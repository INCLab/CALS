# CALS
![Mysql 8.0](https://img.shields.io/badge/mysql-8.0-blue)

### Channel State Information Auto-Labeling System for Large-scale Deep Learning-based Wi-Fi Sensing

> **Abstract:** *Wi-Fi Sensing, which uses Wi-Fi technology to sense the surrounding environments, 
> has strong potentials in a variety of sensing applications. Recently several advanced deep learning-based 
> solutions using CSI (Channel State Information) data have achieved high performance, 
> but it is still difficult to use in practice without explicit data collection, 
> which requires expensive adaptation efforts for model retraining. In this study, 
> we propose a Channel State Information Automatic Labeling System (CALS) that automatically 
> collects and labels training CSI data for deep learning-based Wi-Fi sensing systems. 
> The proposed system allows the CSI data collection process to efficiently collect labeled 
> CSI for labeling for supervised learning using computer vision technologies such as object detection algorithms. 
> We built a prototype of CALS to demonstrate its efficiency and collected data to train deep learning models 
> for detecting the presence of a person in an indoor environment, showing to achieve an accuracy of over 90% 
> with the auto-labeled data sets generated by CALS.*
### CALS Flowchart
<p align="center"><img src="https://user-images.githubusercontent.com/51084152/231077556-d091eb24-0ebd-41dd-9257-8ff6f2cb4dfd.png"  width="400" height="400"/></p>   

### CALS Architecture
<p align="center"><img src="https://user-images.githubusercontent.com/51084152/231078252-398a1f02-095d-4fe3-85a0-dc7e9856a35b.png"  width="500" height="300"/></p>
  
- There are two versions included: a CSV version that works with pre-extracted CSI data in CSV format, and a DB version that collects CSI data directly from the Raspberry Pi extractor and stores it in a MySQL database for further process.
---
## Prerequisites

 - Python 3.x
 - [Nexmon CSI Extractor](https://github.com/seemoo-lab/nexmon_csi) Raspberry Pi B3+/B4(Wi-Fi chip: bcm43455c0) 
 - Install [ByteTrack](https://github.com/ifzhang/ByteTrack) on the server
 - MySQL server
---

## Installation

###  *Server*
1. Clone the repository
```
git clone https://github.com/INCLab/CALS.git
```

2. Install the required dependencies
```
pip install -r requirements.txt
```

3. Change to the project directory
```
cd bytetrack
```

4. Install the [ByteTrack](https://github.com/ifzhang/ByteTrack)  
- *Note: If you want to use another CV module, you can install the desired module. And additionally, columns of output should be modified as follows:*
`time frame_index label`


### *Extractor*
1. Clone the repository
```
git clone https://github.com/INCLab/CALS.git
```

2. Install the  [Nexmon CSI Extractor](https://github.com/seemoo-lab/nexmon_csi)
---

## Usage - CSV
### 1. Time Syncronization  
After installing Nexmon firmware on Raspberry Pi, synchronize time with the camera server.  
>*Method: Network Time Protocol (NTP)*  
>*One of the easiest ways to synchronize time between two devices is to use the Network Time Protocol (NTP). This protocol allows devices to synchronize their clocks over a network connection.*

### 2. Collect CSI data & record video


### 3. 

---
## Referenced Projects

This project takes inspiration from the following two open-source projects:

1. **Nexmon**: The Nexmon project provides firmware patches for collecting CSI on Broadcom Wi-Fi chips. For more information about this project, please visit the [Nexmon GitHub repository](https://github.com/seemoo-lab/nexmon_csi).

2. **ByteTrack**: ByteTrack offers an efficient algorithm for real-time multi-object tracking. For more information about this project, please visit the [ByteTrack GitHub repository](https://github.com/ifzhang/ByteTrack).
