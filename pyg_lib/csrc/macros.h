#pragma once

#ifdef _WIN32
#if defined(pyg_EXPORTS)
#define PYG_API __declspec(dllexport)
#else
#define PYG_API __declspec(dllimport)
#endif
#else
#define PYG_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define PYG_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define PYG_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define PYG_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
