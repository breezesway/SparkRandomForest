import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.text.SimpleDateFormat
import java.util.Date

object PricePaidRecords extends Serializable {
  def main(args: Array[String]): Unit = {
    val time1 = System.currentTimeMillis()
    //val date1 = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date)
    //创建SparkSession(是对SparkContext的包装和增强)
    @transient val spark: SparkSession = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[1]")
      //.master("spark://10.102.65.222:7077")
      //.config("spark.executor.memory","3g")
      //.config("spark.jars","E:\\IDEAWorkP\\PricePaidRecords\\out\\artifacts\\PricePaidRecords_jar\\PricePaidRecords.jar")
      .getOrCreate()

    val time2 = System.currentTimeMillis()

    val df:DataFrame = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("E:\\SparkDatasets\\price_paid_records.csv")
    //E:\SparkDatasets\price_paid_records.csv

    val time3 = System.currentTimeMillis()

    df.printSchema()
    df.show(10)
    val df3 = df.drop("Transaction unique identifier")      //        192.168.149.128  /129 /131
    val df2 = df3.drop("Date of Transfer")
    df2.show(10)

    println(df2.count())
    df2.describe("Price").show()

    val Array(train,test) = df2.randomSplit(Array(0.8,0.2))
    val traindf = train.withColumnRenamed("Price","label")
    val indexer = new StringIndexer().setInputCol("Duration").setOutputCol("Duration_")
    val indexer2 = new StringIndexer().setInputCol("Property Type").setOutputCol("Property Type_")
    val indexer3 = new StringIndexer().setInputCol("Old/New").setOutputCol("Old/New_")
    val indexer4 = new StringIndexer().setInputCol("Town/City").setOutputCol("Town/City_")
    val indexer5 = new StringIndexer().setInputCol("District").setOutputCol("District_")
    val indexer6 = new StringIndexer().setInputCol("County").setOutputCol("County_")
    val indexer7 = new StringIndexer().setInputCol("PPDCategory Type").setOutputCol("PPDCategory Type_")
    val indexer8 = new StringIndexer().setInputCol("Record Status - monthly file only").setOutputCol("Record Status - monthly file only_")
    val assembler = new VectorAssembler().setInputCols(Array("Property Type_", "Old/New_", "Duration_", "Town/City_", "District_", "County_", "PPDCategory Type_", "Record Status - monthly file only_")).setOutputCol("features")
    val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features").setMaxBins(1170).setNumTrees(20)
    val pipeline = new Pipeline().setStages(Array(indexer, indexer2,indexer3,indexer4,indexer5,indexer6,indexer7,indexer8,assembler, rf))

    val time4 = System.currentTimeMillis()

    val model = pipeline.fit(traindf)

    val time5 = System.currentTimeMillis()

    val labelsAndPredictions = model.transform(test)
    labelsAndPredictions.select("prediction").show(10)

    val time6 = System.currentTimeMillis()

    val readtime:Double=(time3-time2)/1000.0
    val fittime:Double=(time5-time4)/1000.0
    val totaltime:Double=(time6-time1)/1000.0
    println("读取数据耗时："+readtime+"s")
    println("模型训练耗时："+fittime+"s")
    println("总耗时："+totaltime+"s")
  }
}
