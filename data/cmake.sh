cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON3_EXECUTABLE=/home/sdjksdafji/PyEnv/cv/bin/python3 \
-D PYTHON3_INCLUDE_DIR=/home/sdjksdafji/PyEnv/cv/include/python3.6m \
-D PYTHON3_LIBRARY=/home/sdjksdafji/PyEnv/cv/lib/python3.6 \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/home/sdjksdafji/PyEnv/cv/lib/python3.6/site-packages/numpy/core/include \
-D PYTHON3_PACKAGES_PATH=/home/sdjksdafji/PyEnv/cv/lib/python3.6/site-packages \
-D BUILD_EXAMPLES=ON ..
