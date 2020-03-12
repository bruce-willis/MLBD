def objective_xgb(space):
    import os
    import sys

    import pyspark
    import pyspark.sql.functions as F
    from pyspark.conf import SparkConf
    from pyspark.sql import SQLContext
    from pyspark.sql import SparkSession
    from pyspark.sql import Row

    COMMON_PATH = '/workspace/common'

    sys.path.append(os.path.join(COMMON_PATH, 'utils'))

    os.environ['PYSPARK_SUBMIT_ARGS'] = """
--jars {common}/xgboost4j-spark-0.72.jar,{common}/xgboost4j-0.72.jar
--py-files {common}/sparkxgb.zip pyspark-shell
""".format(common=COMMON_PATH).replace('\n', ' ')


    spark = SparkSession \
                    .builder \
                    .master('local[*]') \
                    .appName("spark_sql_examples") \
                    .config("spark.executor.memory", "12g") \
                    .config("spark.driver.memory", "12g") \
                    .config("spark.task.cpus", "8") \
                    .config("spark.executor.cores", "8") \
                    .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    from metrics import rocauc, logloss, ne, get_ate
    from processing import split_by_col
    from sparkxgb.xgboost import XGBoostEstimator
    

    DATA_PATH = '/workspace/data/criteo'
    TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')

    train_df = sqlContext.read.format("com.databricks.spark.csv") \
        .option("delimiter", ",") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load('file:///' + TRAIN_PATH + "_part_train.csv")

    val_df = sqlContext.read.format("com.databricks.spark.csv") \
        .option("delimiter", ",") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load('file:///' + TRAIN_PATH + "_part_valid.csv")

    from pyspark.ml import PipelineModel
    pipeline_model = PipelineModel.load(os.path.join(DATA_PATH, 'pipeline_model'))

    train_df = pipeline_model \
            .transform(train_df) \
            .select(F.col('_c0').alias('label'), 'features', 'id') \
            .cache()

    val_df = pipeline_model \
            .transform(val_df) \
            .select(F.col('_c0').alias('label'), 'features', 'id') \
            .cache()

    estimator = XGBoostEstimator(**space)
    print('SPACE:', estimator._input_kwargs_processed())
    success = False
    attempts = 0
    model = None
    while not success and attempts < 2:
        try:
            model = estimator.fit(train_df)
            success = True
        except Exception as e:
            attempts += 1
            print(e)
            print('Try again')
        
    log_loss = logloss(model, val_df, probabilities_col='probabilities')
    roc_auc = rocauc(model, val_df, probabilities_col='probabilities')
    
    print('*'*20, '\n'*5)
    print('LOG-LOSS: {}, ROC-AUC: {}'.format(log_loss, roc_auc))
    print('\n'*5, '*'*20)

    sc.stop()
    spark.stop()

    from hyperopt import STATUS_OK
    return {'loss': log_loss, 'rocauc': roc_auc, 'status': STATUS_OK }

