from setuptools import setup

package_name = "fracmem_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "fracmem"],
    zip_safe=True,
    maintainer="Adithyan Lalu",
    maintainer_email="adithyanlalu@gmail.com",
    description="ROS2 streaming interface for the fracmem compressed fractional-order derivative filter",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "derivative_node = fracmem_ros2.derivative_node:main",
        ],
    },
)
