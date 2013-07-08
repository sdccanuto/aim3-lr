package com.celebihacker.ml.logreg.mapred;

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
public class GlobalJobSettings {

  // TODO Minor: Remove this redundancy
  static final String CONFIG_FILE_PATH = "core-site-local.xml";
  // static final String CONFIG_FILE_PATH = "core-site-pseudo-distributed.xml";

  static final Level LOG_LEVEL = Level.DEBUG;

  static final String BASE_MODEL_PATH = "sfo-base-model.seq";

  // Can be changed (e.g. by testcase)
   static DatasetInfo datasetInfo = RCV1DatasetInfo.get();

  // --------- Settings for execution in a cluster ------------

  static final String JAR_PATH = "target/aim3-logreg-0.0.1-SNAPSHOT-job.jar";

}
