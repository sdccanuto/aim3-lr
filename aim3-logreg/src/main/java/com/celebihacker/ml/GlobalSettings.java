package com.celebihacker.ml;

import org.apache.log4j.Level;

import com.celebihacker.ml.datasets.DatasetInfo;
import com.celebihacker.ml.datasets.RCV1DatasetInfo;


/**
 * Static settings for the SFO Hadoop jobs.
 * 
 * We can refactor this at a later time to be loaded dynamically on runtime.
 * This would require to pass the arguments to all the tasks (map/reduce) via
 * job configuration, distributed cache or hdfs.
 */
public class GlobalSettings {

  // TODO Minor: Remove this redundancy
  public static final String CONFIG_FILE_PATH = "core-site-local.xml";
  // static final String CONFIG_FILE_PATH = "core-site-pseudo-distributed.xml";

  public static final Level LOG_LEVEL = Level.DEBUG;

  public static final DatasetInfo datasetInfo = RCV1DatasetInfo.get();

  // --------- Settings for execution in a cluster ------------

  public static final String JAR_PATH = "target/aim3-logreg-0.0.1-SNAPSHOT-job.jar";

}
