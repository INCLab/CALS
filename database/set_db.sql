CREATE DATABASE `csi`;
USE `csi_data`;

CREATE TABLE `csi` (
  `time` FLOAT NOT NULL,
  `MAC` CHAR(255) NOT NULL,
  `band` CHAR(128) NOT NULL,
  `bandwidth` CHAR(128) NOT NULL,
  `i_values` FLOAT,
  `q_values` FLOAT,
  PRIMARY KEY(`MAC`)
);
