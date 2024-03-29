package com.celebihacker.ml;

import org.apache.log4j.Level;

/**
 * Static settings, for various Hadoop jobs
 * 
 * The goal is to remove all those global static settings and load them at runtime
 */
public class GlobalSettings {

  public static final String CONFIG_FILE_PATH = "core-site-local.xml";
  // static final String CONFIG_FILE_PATH = "core-site-pseudo-distributed.xml";

  public static final Level LOG_LEVEL = Level.DEBUG;

  // --------- Settings for execution in a cluster ------------
  public static final String JAR_PATH = "target/aim3-logreg-0.0.1-SNAPSHOT-job.jar";

}
