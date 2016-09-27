mkdir build
cd build
mkdir cuhe
cd cuhe
nvcc -std=c++11 ../../cuhe/*.cu -c
cd ..
mkdir DHS
cd DHS
nvcc -std=c++11 ../../examples/DHS/*.cu -c
cd ..
mkdir Prince
cd Prince
nvcc -std=c++11 -lgomp ../../examples/Prince/*.cu -c
cd ..
mkdir tests
cd tests
nvcc -std=c++11 ../../tests/*.cu -c
cd ..
mkdir bin
nvcc -std=c++11 -lm -lntl -lgmp cuhe/*.o tests/test_ModP.o -o bin/test_ModP
nvcc -std=c++11 -lm -lntl -lgmp cuhe/*.o tests/test_ntt.o -o bin/test_ntt
nvcc -std=c++11 -lm -lntl -lgmp cuhe/*.o tests/test_utils.o -o bin/test_utils
nvcc -std=c++11 -lm -lntl -lgmp cuhe/*.o DHS/*.o -o bin/simple_DHS
nvcc -std=c++11 -lm -lntl -lgmp -lgomp cuhe/*.o Prince/*.o -o bin/test_Prince
