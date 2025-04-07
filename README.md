# Run Inference
docker run -v /isis/home/hasana3/ros2_image/workspace_src/:/ros2_ws/src -v /isis/home/hasana3/ByteTrack:/byteTrack_data -v /isis/home/hasana3/ros2_image/output:/output -v /isis/home/hasana3/ros2_image/bagfiles:/bagfiles -it --rm --gpus all --net=host bytetrack-ros2
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch src/perception/perception/launch/test.xml