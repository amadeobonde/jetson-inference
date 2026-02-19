# FindLiburing.cmake
# Locate liburing headers and library via pkg-config or manual search.
#
# Sets:
#   Liburing_FOUND
#   Liburing_INCLUDE_DIRS
#   Liburing_LIBRARIES

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(_LIBURING QUIET liburing)
endif()

find_path(Liburing_INCLUDE_DIR
    NAMES liburing.h
    HINTS ${_LIBURING_INCLUDE_DIRS}
    PATHS /usr/include /usr/local/include
)

find_library(Liburing_LIBRARY
    NAMES uring
    HINTS ${_LIBURING_LIBRARY_DIRS}
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Liburing
    REQUIRED_VARS Liburing_LIBRARY Liburing_INCLUDE_DIR
)

if(Liburing_FOUND)
    set(Liburing_INCLUDE_DIRS ${Liburing_INCLUDE_DIR})
    set(Liburing_LIBRARIES ${Liburing_LIBRARY})

    if(NOT TARGET Liburing::Liburing)
        add_library(Liburing::Liburing UNKNOWN IMPORTED)
        set_target_properties(Liburing::Liburing PROPERTIES
            IMPORTED_LOCATION "${Liburing_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Liburing_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(Liburing_INCLUDE_DIR Liburing_LIBRARY)
