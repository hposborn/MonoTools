#Your .theanorc file may need some bells and whistles to allow exoplanet to work (I sure know mine did).

[global]
#As I'm frequently running multiple models at once, I limit it to cpu
device=cpu
#This maybes float arrays uniforms (we probably dont need 64bit, but you might)
floatX=float32
#If you're using a non-default version of gcc/g++, note that here:
#cxx=/usr/local/Cellar/gcc/9.3.0_1/bin/g++-9
#-fbracket-depth=1024 allow a need a big model to be created with <1024 parameters
#-Wno-c++11-narrowing allows some C++11 functionality on non-C++11 models.
gcc.cxxflags="-fbracket-depth=1024 -Wno-c++11-narrowing"
