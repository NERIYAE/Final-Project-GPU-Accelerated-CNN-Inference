#include <cuda_runtime.h>
#include "cuconv_api.h"


__global__ void scalar_prods_kernel(     //הגדרה של הקרנל הראשון
    const float* __restrict__ input,      //מצביע על הקלט לקריאה בלבד ורמז לקומפיילר שאין חפיפה בין הכתובות לאופטמיזציה
    const float* __restrict__ filters,   
    float* __restrict__ partial_outputs, 
    int H, int W,
    int pad_h, int pad_w,
    int filterH, int filterW,
    int depth, int numFilters,
    int outH, int outW)
{
    int fyfx = blockIdx.z;                      //   לקיחת המיקום מתוך כל מיקומי הפילטר שהוגדר בהשקה בגריד ופירוקם למיקום דו מימדי      
    if (fyfx >= filterH * filterW) return;      // וידוי שלא יתווצרו עוד בלוקים מיותרים
    int fy = fyfx / filterW;
    int fx = fyfx % filterW;

    int filterIdx = blockIdx.y;                //  הגדרת הפילטר מתוך כמות הפילטרים שנשלחו שהוגדרו בלוק 
    if (filterIdx >= numFilters) return;       // בדיקה שלא יתווצרו יותר בלוקים בציר ואי לטיפול בפילטר

    int outPixelIdx = blockIdx.x * blockDim.x + threadIdx.x;  // יצירת הגדרת ת'רדים חד מימדים
    if (outPixelIdx >= outH * outW) return;     // וידוי שלא יופעלו ת'רדים מיותרים

    int outY = outPixelIdx / outW;             // הוצאת מיקום הפלט עבור ת'רד יחיד
    int outX = outPixelIdx % outW;

    
    int inY = outY - pad_h + fy;              //הוצאת מיקום הקלט עבור ת'רד יחיד
    int inX = outX - pad_w + fx;

    float sum = 0.0f;                                                    
    if ((unsigned)inY < (unsigned)H && (unsigned)inX < (unsigned)W) {   // בדיקה שת'רדים לא מחשבים סתם מיקום מהקלט
        for (int d = 0; d < depth; d++) {                               //כל ת'רד אחראי על עומק של הפילטר והקלט               
            int inIdx = d * H * W + inY * W + inX;                                 //יצירת אינדקס חד מימדי למיקום על הקלט לת'רדים
            int fOff  = (((filterIdx * filterH + fy) * filterW + fx) * depth) + d;     //  יצירת אינדקס חד מימדי למיקום של המשקולת בפילטר
            float xv = input[inIdx];
            float wv = filters[fOff];
            sum += xv * wv;
        }
    } else {
        sum = 0.0f;
    }

    int partialIdx = ( (filterIdx * (filterH * filterW) + fyfx) * outH * outW ) + outPixelIdx;    // יצירת אינדקס חד מימדי לת'רדים לשמירת התוצרים החלקיים
    partial_outputs[partialIdx] = sum;
}


__global__ void sum_kernel(                                          //הגדרה של הפילטר השני
    const float* __restrict__ partial_outputs,                       //  מצביע על מערך התוצרים החלקיים לקריאה בלבד ורמז לקומפיילר שאין חפיפות           
    float* __restrict__ output,                                     //   מצביע לכתובת שבא נמצא מערך התוצרים הסופיים
    int outH, int outW,
    int numFilters, int kernelSize       
){
    int filterIdx   = blockIdx.y;                              
    if (filterIdx >= numFilters) return;  

    int outPixelIdx = blockIdx.x * blockDim.x + threadIdx.x;           //מגדירים לכל ת'רד על איזה פיקסל לעבוד
    if (outPixelIdx >= outH * outW) return;    

    float sum = 0.0f;
    for (int k = 0; k < kernelSize; k++) {                                  // סכימה לכל ההיסטים של הפילטר
        int pIdx = ((filterIdx * kernelSize) + k) * outH * outW + outPixelIdx;      // יצירת אינדרקס חד מימדי ללקיחת תוצר חלקי של מכפלות העומק
        sum += partial_outputs[pIdx];
    }
    int outIdx = filterIdx * outH * outW + outPixelIdx;              // יצירת טינדקס חד מימדי לסידור פיקסלי הפלט הסופיים של הקונבולוציה
    output[outIdx] = sum;   // תוצר הקונבולוציה
}

//  API 
extern "C" int cuconv_conv_forward(                // פונקציה חיצונית שניתן לגשת אליה גם מחוץ לקוד הזה וגם כבר הכרזתי עליה
    const float* x, const float* w, float* y,
    int N, int C, int H, int W,
    int M, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int groups)
{
    if (N != 1 || groups != 1) return -1;              // הגדרת גבולות פעולה של האלגורתם
    if (stride_h != 1 || stride_w != 1) return -2;
    if (dil_h != 1 || dil_w != 1) return -3;

    int Ho = H + 2*pad_h - R + 1;              // הגדרת גובה הפלט הסופי
    int Wo = W + 2*pad_w - S + 1;

    const int kernelSize = R * S;                 
    const bool is1x1 = (kernelSize == 1);         // הגדרת משתנה מסוג בול כדי לטפל נקודתית במקרה של פילטר 1

    float* d_partial = nullptr;                           // הגדרת מצביע של מערך התוצרים החלקיים
    if (!is1x1) {                                 
        size_t partial_sz = (size_t)kernelSize * M * Ho * Wo;        //אם אנחנו במקרה כללי אז ניצור גודל מתאים להקצאת זיכרון על המכשיר לתוצרים החלקיים
        cudaMalloc(&d_partial, partial_sz * sizeof(float));
    }
    float* out_or_partial = is1x1 ? y : d_partial;      //   יצירה של מצביע שאם אנחנו במקרה כללי אז הוא יהיה הכתובת של המצביע של התוצרים החלקיים 

    
    {
        int threads = 256;                                // הגדרת כמות הת'רדים בבלוק
        int blocks_x = (Ho * Wo + threads - 1) / threads;            // בגלל אינט אז נמנע עיגול של בלוקים בציר איקס אז נעגל ללמעלה
        dim3 grid(blocks_x, M, kernelSize);                        //   הגדרת הגריד
        scalar_prods_kernel<<<grid, threads>>>(                       //    השקת הקרנל
            x, w, out_or_partial,
            H, W,
            pad_h, pad_w,
            R, S,
            C, M,
            Ho, Wo
        );
        cudaGetLastError();               // פונקציה מהספריה של קודה שמחזירה את השגיאה האחרונה שהייתה
    }

   
    if (!is1x1) {                          //      במידה ואנחנו במצב הכללי אז המשך בקוד  
        int threads = 256;
        int blocks_x = (Ho * Wo + threads - 1) / threads;
        dim3 grid(blocks_x, M, 1);            //    כאן נגדיר את ציר זאד ל1
        sum_kernel<<<grid, threads>>>(
            d_partial, y, Ho, Wo, M, kernelSize
        );
        cudaGetLastError();
    }                                           

    if (!is1x1) cudaFree(d_partial);     //   אם אנחנו במקרה הכללי אז שחרר את הזיכרון שבמכשיר ששמר את התוצרים החלקיים          
    return 0;
}

