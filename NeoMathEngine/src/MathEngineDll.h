/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <cstdint>

#include <NeoMathEngine/NeoMathEngineDefs.h>

#if FINE_PLATFORM( FINE_WINDOWS )
#include <windows.h>
#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#include <dlfcn.h>
#endif

#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// A dynamic link library
class CDll : public CCrtAllocatedObject {
public:
	CDll() = default;
	virtual ~CDll() { Free(); }

	// Loads the library
	bool Load( const char* fileName );

	// Checks if the library is loaded
	bool IsLoaded() const { return handle != nullptr; }

	// Gets the function address in the library. Cast to function pointer type T
	template <typename T>
	T GetProcAddress( const char* functionName ) const;

	// Unloads the library
	void Free();

private:
	void* handle{};
};

inline bool CDll::Load( const char* fileName )
{
#if FINE_PLATFORM( FINE_WINDOWS )
	handle = ::LoadLibraryExA( fileName, 0, LOAD_WITH_ALTERED_SEARCH_PATH );
#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
	handle = ::dlopen( fileName, RTLD_LAZY | RTLD_GLOBAL );
#else
	#error "Platform is not supported!"
#endif
	return ( handle != nullptr );
}

template <typename T>
inline T CDll::GetProcAddress( const char* functionName ) const
{
#if FINE_PLATFORM( FINE_WINDOWS )
	return reinterpret_cast<T>( reinterpret_cast<uintptr_t>( ::GetProcAddress( static_cast<HMODULE>( handle ), functionName ) ) );
#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
	return reinterpret_cast<T>( reinterpret_cast<uintptr_t>( ::dlsym( handle, functionName ) ) );
#else
	#error "Platform is not supported!"
#endif
}

inline void CDll::Free()
{
	if( handle == 0 ) {
		return;
	}

#if FINE_PLATFORM( FINE_WINDOWS )
	::FreeLibrary( static_cast<HMODULE>( handle ) );
#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
	::dlclose( handle );
#else
	#error "Platform is not supported!"
#endif
	handle = 0;
}

} // namespace NeoML
