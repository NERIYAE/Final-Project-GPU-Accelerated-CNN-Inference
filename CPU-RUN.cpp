#include <cstdio>
#include <ctime>      
#include <vector>
#include <random>
using namespace std;


struct Config {               // הגדרת מבנה 
    const char* table; 
    const char* label; 
    int N, H, R, M, C; 
};


static inline double now_us(){                // פונקציה ללקיחת זמן בלינוקס בתקווה שהקומפיילר לא יגרום לעצירה בזמן ריצה אלא שהקוד שלה יהיה כתוב ישר כשצריך אותה למניעת עיכובים
    timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);     //  לקיחת הזמן והשמה של הערך הנמדד
    return (double)ts.tv_sec*1e6 + (double)ts.tv_nsec/1e3;   // החזרת זמן בדיוק של עד ננו שניות
}


size_t inIdx(int c, int y, int x, int H, int W) {                       // אינדקסי עזר
    return ((size_t)c * (size_t)H + (size_t)y) * (size_t)W + (size_t)x;   // החזרה של ערך חד מימדי בערכי הקלט
}

size_t filtIdx_cuda(int m, int c, int fy, int fx, int C, int R, int S) {
    return ((((size_t)m * (size_t)R + (size_t)fy) * (size_t)S + (size_t)fx) * (size_t)C + (size_t)c);  // M×R×S×C חישוב אינדקס חד ממדי לפילטר לפי הסדר 
}

size_t partIdx_cuda(int k, int m, int y, int x, int M, int Ho, int Wo, int K) {
    return (((size_t)m * (size_t)K + (size_t)k) * (size_t)Ho + (size_t)y) * (size_t)Wo + (size_t)x;  // mxKxHoxWo החזרה של ערך חד מימדי לתוצרים החלקיים לפי 
}

size_t outIdx(int m, int y, int x, int Ho, int Wo) {
    return ((size_t)m * (size_t)Ho + (size_t)y) * (size_t)Wo + (size_t)x;  // mxHoxWo  החזרה של ערך חד מימדי לפיקסלים בפלט לפי
}


static void fill_random(vector<float>& v){
    mt19937 rng(42);
    uniform_real_distribution<float> U(-1.f, 1.f); 
    for (auto& x : v) x = U(rng);
}

static void computeScalarProducts(    //  פונקציה מקומית לביצוע התוצרים החלקיים
    const float* __restrict__ input,          
    const float* __restrict__ filters,        
    float* __restrict__ partial,           
    int H, int W,
    int Ho, int Wo,
    int pad_h, int pad_w,
    int C, int M,
    int R, int S)
{
    const int K = R*S;                  
    for (int fy=0; fy<R; ++fy){                    // ריצה טורית עם לולאה מקוננת לאורך ורוחב היסטי הפילטר
        for (int fx=0; fx<S; ++fx){                    // 
            const int k = fy*S + fx;                       // הגדרת ההיסט
            for (int m=0; m<M; ++m){                         // לפי כמות הפילטרים 
                for (int y=0; y<Ho; ++y){                       // עוברים על הפלט
                    for (int x=0; x<Wo; ++x){                      //
                        float acc = 0.0f;
                        const int inY = y - pad_h + fy;            // הקלט עם הפאדינג והמיקום הטורי של היסט הפילטר
                        const int inX = x - pad_w + fx;
                        if ((unsigned)inY < (unsigned)H && (unsigned)inX < (unsigned)W){        // בדיקת גבולות הקלט
                            for (int c=0; c<C; ++c){                        // עומק הפילטר והקלט
                                float xv = input[  inIdx(c,inY,inX,H,W) ];         // קח ערך לפי האינדוקס שיבוצע אל הקלט
                                float wv = filters[ filtIdx_cuda(m,c,fy,fx,C,R,S) ];  // קח ערך לפי האינדוקס שיבוצע אל הפילטר כלומר משקל
                                acc += xv * wv;                                    // צור תוצר חלקי וסכום לאורך כל העומק
                            }
                        }
                        partial[ partIdx_cuda(k,m,y,x,M,Ho,Wo,K) ] = acc;       // הצבה באינדקס המתאים
                    }
                }
            }
        }
    }
}

static void sumPartialResults(             //  פונקציה מקומית לסכימת התוצרים החלקיים
    const float* __restrict__ partial,
    float* __restrict__ output,         
    int Ho, int Wo,
    int M, int K)
{
    for (int m=0; m<M; ++m){         // מעבר על כמות הפילטרים
        for (int y=0; y<Ho; ++y){           // מעבר על שורות ועמודות הפלט
            for (int x=0; x<Wo; ++x){          
                float s = 0.0f;
                for (int k=0; k<K; ++k)                    
                    s += partial[ partIdx_cuda(k,m,y,x,M,Ho,Wo,K) ];        // לאחר מעבר על כל ההיסטים האפשריים סכימה והוספה כמו השלב האחרון בפעולת הקונבולוציה
                output[ outIdx(m,y,x,Ho,Wo) ] = s;     //  הצבה במיקום המתאים בפלט הסופי של הקונבולוציה לאחר אינדוקס מתאים
            }
        }
    }
}


static float run(const Config& cfg, int inner_runs) {      
    int N = cfg.N, C = cfg.C, H = cfg.H, W = cfg.H;
    int M = cfg.M, R = cfg.R, S = cfg.R;
    int pad = (R - 1) / 2, stride = 1, dil = 1, groups = 1;


    size_t xin  = (size_t)N * C * H * W;          //
    size_t win  = (size_t)M * C * R * S;
    int Ho = H + 2 * pad - R + 1;      //
    int Wo = W + 2 * pad - S + 1;
    size_t yout = (size_t)N * M * Ho * Wo;      //

    vector<float> hx(xin), hw(win), hy(yout, 0.0f);       //
    vector<float> partial((size_t)(R*S) * M * Ho * Wo);        //

    fill_random(hx); //
    fill_random(hw);

    double t0 = now_us();                          // לקיחת זמן התחלה
    for (int i = 0; i < inner_runs; i++) {                     // ריצה של כמות הפעמים שהוגדרה
    computeScalarProducts(hx.data(), hw.data(), partial.data(),
                          H, W, Ho, Wo, pad, pad, C, M, R, S);
    sumPartialResults(partial.data(), hy.data(), Ho, Wo, M, R*S);
    }
    double t1 = now_us();  // לקיחת זמן סוף

    double ms = double((t1 - t0) / inner_runs / 1000.0f);       // הסקת זמן ממוצע לפעולה
    printf("%s %s: N=%d, HxW=%dx%d, R=S=%d, M=%d, C=%d -> Ho×Wo=%dx%d | Avg=%.3f ms\n",
    cfg.table, cfg.label, N, H, W, R, M, C, Ho, Wo, ms);
    return ms;                                           
                                         
}

int main(){                 //
    int INNER = 20;
    vector<Config> cases = {
        {"T3-1x1","A", 1,  7,1,  256, 832},
        {"T3-1x1","B", 1, 14,1, 1024, 256},
        {"T3-1x1","C", 1, 27,1,  256,  64},
        {"T4-3x3","A", 1,  4,3,  384, 192}, 
        {"T4-3x3","B", 1, 13,3,  384, 384},
        {"T5-5x5","A", 1,  7,5,  128,  48},
    };

    printf("Each configuration runs INNER=%d iterations averaged.\n", INNER);
    for (const auto& cfg : cases) 
        run(cfg, INNER);
    return 0;
}