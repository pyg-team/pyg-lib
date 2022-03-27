#pragma once

#include "macros.h"
#include <cstdint>

namespace pyg {

PYG_API int64_t cuda_version();

namespace detail {

extern "C" PYG_INLINE_VARIABLE auto _register_ops = &cuda_version;
#ifdef HINT_MSVC_LINKER_INCLUDE_SYMBOL
#pragma comment(linker, "/include:_register_ops")
#endif

} // namespace detail
} // namespace pyg
