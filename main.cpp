#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "cuconv_api.h"

using namespace std;


struct Config {                         // הגדרת המבנה של ווקטור הניסויים         
    const char* table; 
    const char* label; 
    int N, H, R, M, C; 
};

static void fill_random(vector<float>& v) {           //פונקציה מקומית למילוי רנדומלי של ערכים לקלט ולמשקולות של הפילטר
    mt19937 rng(42);
    uniform_real_distribution<float> U(-1.f, 1.f);
    for (auto& x : v) x = U(rng);
}


static float run(const Config& cfg, int inner_runs) {            //פונקציה מקומית להרצה שמקבלת מספר הרצות אובייקט יחיד מהמבנה של הניסוי
    int N = cfg.N, C = cfg.C, H = cfg.H, W = cfg.H;              //לקיחת נתונים ממערך האובייקט לשים לב לסימטריות
    int M = cfg.M, R = cfg.R, S = cfg.R;
    int pad = (R - 1) / 2, stride = 1, dil = 1, groups = 1;


    size_t xin = (size_t)N * C * H * W;                        // גודל המערך של הקלט בזיכרון שצריך להיות מוגדר
    size_t win = (size_t)M * C * R * S;                        // המשקולות של הפילטר
    int Ho = H + 2 * pad - R + 1;                              // גודל גובה הפלט בהתחשבות בפדינג
    int Wo = W + 2 * pad - S + 1;
    size_t yout = (size_t)N * M * Ho * Wo;                     // גודל המערך של הפלט

    vector<float> hx(xin), hw(win), hy(yout, 0.0f);             //הגדרה של מערכים על זיכרון המעבד בגדלים שהוגדרו לפני
    fill_random(hx); fill_random(hw);              

    float* dx = nullptr, * dw = nullptr, * dy = nullptr;            //הגדרת מצביעים למכשיר לקלט משקולות ופלט
    cudaMalloc(&dx, xin * sizeof(float));                           // הקצאת מקום בזיכרון המכשיר והעברת כתובת תחילת הזיכרון למצביע שהוגדר לפני
    cudaMalloc(&dw, win * sizeof(float));
    cudaMalloc(&dy, yout * sizeof(float));
    cudaMemcpy(dx, hx.data(), xin * sizeof(float), cudaMemcpyHostToDevice);  //העתקת הזיכרון מהמארח לכתובת המוצבעת במכשיר בגודל שהוקצאה לפני
    cudaMemcpy(dw, hw.data(), win * sizeof(float), cudaMemcpyHostToDevice);



    //cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);     
    //cudaEventRecord(start);
    for (int i = 0; i < inner_runs; i++) {                                                                
        cuconv_conv_forward(dx, dw, dy, N, C, H, W, M, R, S, pad, pad, stride, stride, dil, dil, groups);
    }
    //cudaEventRecord(stop); cudaEventSynchronize(stop);
    //float ms = 0.f; cudaEventElapsedTime(&ms, start, stop);
    //ms /= inner_runs;

    //printf("%s %s: N=%d, HxW=%dx%d, R=S=%d, M=%d, C=%d -> Ho×Wo=%dx%d | Avg=%.3f ms\n",
        //cfg.table, cfg.label, N, H, W, R, M, C, Ho, Wo, ms);

    // bring back one result buffer to avoid DCE if i want to know for sure that there was a calclulting
    //cudaMemcpy(hy.data(), dy, yout * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dx); cudaFree(dw); cudaFree(dy);           // שחרור הזיכרון במכשיר 
    //cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

//static void wait_for_user() {                                          
  //  printf(">> Press ENTER to continue to next configuration...\n");
    //fflush(stdout);
    //getchar(); // כל לחיצה על ENTER ממשיכה
//}
       
int main() {                         //
    int INNER = 20; 
    vector<Config> cases = {         //יצירת ווקטור מהסוג שהוגדר שמכיל רשימת תצורות
       
        // {"T3-1x1","A", 1,  7,1,  256, 832},
        //{"T3-1x1","B", 1, 14,1, 1024, 256},
        //{"T3-1x1","C", 1, 27,1,  256,  64},
        //{"T4-3x3","A", 1,  4,3,  384, 192}, 
        //{"T4-3x3","B", 1, 13,3,  384, 384},
        //{"T5-5x5","A", 1,  7,5,  128,  48},
		
		{"T4-3x3","E1", 1,  32,3,  64,  64},
        {"T4-3x3","E2", 1,  32,3, 128, 128},
        {"T4-3x3","E3", 1,  64,3, 128, 128},
        {"T4-3x3","E4", 1,  64,3, 256, 256},
    };
    //printf("Each configuration runs INNER=%d iterations averaged, then waits for ENTER.\n", INNER);


    for (const auto& cfg : cases) {                       //עבור כל אובייקט שבמערך מסוג שהוגדר בצע את הלולאה עד שנגמר האובייקטים במערך
        run(cfg, INNER);
       // wait_for_user();
    }
    return 0;
}

