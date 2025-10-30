#pragma once

                                                  // ייצוא או יבוא של פונקציות 
#ifdef CUCONV_EXPORTS                             // הוראה לקומפיילר לייצא או ליבא קובץ ספריה במידה וקיים
  #define CUCONV_API __declspec(dllexport)
#else
  #define CUCONV_API __declspec(dllimport)
#endif

extern "C" CUCONV_API int cuconv_conv_forward(     // הגדרת הפונקציה כחיצונית עם המשתנים שהיא אמורה לקבל
    const float* x, const float* w, float* y,
    int N, int C, int H, int W,
    int M, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int groups
);
