objects = Mandelbrot.o TGA.o

all: $(objects)
	nvcc -arch=sm_20 $(objects) -o mandelbrot

%.o: %.cpp
	nvcc -std=c++11 -x cu -arch=sm_20 -I. -dc $< -o $@

%.o: %.cu
	nvcc -std=c++11 -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o mandelbrot

