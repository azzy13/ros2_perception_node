# Run Inference
Run the docker
docker run -v /isis/home/hasana3/ros2_image/workspace_src/:/ros2_ws/src -v /isis/home/hasana3/ByteTrack:/byteTrack_data -v /isis/home/hasana3/ros2_image/output:/output -v /isis/home/hasana3/ros2_image/bagfiles:/bagfiles -it --rm --gpus all --net=host bytetrack-ros2

Build
colcon build

Source files
source /opt/ros/humble/setup.bash
source install/setup.bash

Launch the perception node
ros2 launch src/perception/perception/launch/test.xml