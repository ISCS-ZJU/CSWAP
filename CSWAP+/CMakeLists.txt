cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)
set(CMAKE_PREFIX_PATH /home/pytorch/pytorch/torch/share/cmake)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

add_subdirectory(zfp)
set (EXTRA_LIBS ${EXTRA_LIBS} zfp)

add_executable(vgg16 vgg16-cifar10.cpp)
target_link_libraries(vgg16 "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET vgg16 PROPERTY CXX_STANDARD 14)
#target_link_directories(vgg16 /usr/local/cuda/lib64)

add_executable(cp_vgg16 cp-vgg16.cpp)
target_link_libraries(cp_vgg16 "${TORCH_LIBRARIES}")
set_property(TARGET cp_vgg16 PROPERTY CXX_STANDARD 14)
#target_link_directories(cp_vgg16 /usr/local/cuda/lib64)

add_executable(vgg19 vgg19-cifar10.cpp)
target_link_libraries(vgg19 "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET vgg19 PROPERTY CXX_STANDARD 14)

#add_executable(lenet lenet-cifar10.cpp)
#target_link_libraries(lenet "${TORCH_LIBRARIES}")
#set_property(TARGET lenet PROPERTY CXX_STANDARD 14)

add_executable(alexnet alexnet-cifar10.cpp)
target_link_libraries(alexnet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
#set_property(TARGET alexnet PROPERTY CXX_STANDARD 14)

add_executable(resnet resnet-cifar10.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET resnet PROPERTY CXX_STANDARD 14)

add_executable(desnet Desnet-cifar10.cpp)
target_link_libraries(desnet "${TORCH_LIBRARIES}")
set_property(TARGET desnet PROPERTY CXX_STANDARD 14)

add_executable(inception inception-v3-cifar100.cpp)
target_link_libraries(inception "${TORCH_LIBRARIES}")
set_property(TARGET inception PROPERTY CXX_STANDARD 14)

#add_executable(tinyimagenet tinyimagenettest.cpp tinyimagenet.h)
#target_link_libraries(tinyimagenet "${TORCH_LIBRARIES}")
#set_property(TARGET tinyimagenet PROPERTY CXX_STANDARD 14)
#target_link_directories(tinyimagenet /usr/local/cuda/lib64)

add_executable(alexnet-tinyimagenet AlexNet-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(alexnet-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET alexnet-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(mobilenet-tinyimagenet mobilenet-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(mobilenet-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET mobilenet-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(plain20-tinyimagenet plain20-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(plain20-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET plain20-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(vgg16-tinyimagenet vgg16-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(vgg16-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET vgg16-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(vgg19-tinyimagenet vgg19-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(vgg19-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET vgg19-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(resnet-tinyimagenet resnet-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(resnet-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET resnet-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(squeezenet-tinyimagenet squeezenet-tinyimagenet.cpp tinyimagenet.h)
target_link_libraries(squeezenet-tinyimagenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET squeezenet-tinyimagenet PROPERTY CXX_STANDARD 14)

add_executable(mobilenet mobilenet-cifar10.cpp)
target_link_libraries(mobilenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET mobilenet PROPERTY CXX_STANDARD 14)
#target_link_directories(mobilenet /usr/local/cuda/lib64)

add_executable(plain20 plain20-cifar10.cpp)
target_link_libraries(plain20 "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET plain20 PROPERTY CXX_STANDARD 14)
#target_link_directories(plain20 /usr/local/cuda/lib64)

add_executable(squeezenet squeezenet-cifar10.cpp)
target_link_libraries(squeezenet "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET squeezenet PROPERTY CXX_STANDARD 14)

add_executable(bert bert.cpp)
target_link_libraries(bert "${TORCH_LIBRARIES}" ${EXTRA_LIBS})
set_property(TARGET bert PROPERTY CXX_STANDARD 14)

add_executable(simple simple.c)
target_link_libraries(simple "${TORCH_LIBRARIES}" ${EXTRA_LIBS} m stdc++)
#set_property(TARGET simple PROPERTY CXX_STANDARD 14)

#add_executable(process process.cpp)
#target_link_libraries(process "${TORCH_LIBRARIES}")
#set_property(TARGET process PROPERTY CXX_STANDARD 11)

add_definitions("-g")


