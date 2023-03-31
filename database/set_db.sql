CREATE DATABASE `csi`;
USE `csi_data`;

CREATE TABLE `csi` (
  `time` FLOAT NOT NULL,
  `MAC` varchar(512) PRIMARY NOT NULL,
  `band` varchar NOT NULL,
  `bandwidth` varchar NOT NULL,
  `values` TIMESTAMP DEFAULT (current_timestamp) on update current_timestamp
);
