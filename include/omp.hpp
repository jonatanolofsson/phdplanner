#pragma once

#ifdef NOPAR
    #define PARFOR
    #define PARSECS
    #define PARSEC
#else
    #define PARFOR _Pragma("omp parallel for")
    #define PARSECS _Pragma("omp parallel sections")
    #define PARSEC _Pragma("omp section")
#endif

