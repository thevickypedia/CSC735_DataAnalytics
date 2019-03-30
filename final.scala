// Databricks notebook source
//Loading in Data
import org.apache.spark.sql.functions._

val arabica = spark.read.option("header", "true")  
  .option("inferSchema","true")
  .option("multiLine", "true")
  .option("mode", "DROPMALFORMED")
  .csv("/FileStore/tables/arabica_data_cleaned.csv")


val robusta = spark.read.option("header", "true")  
  .option("inferSchema","true")
  .option("multiLine", "true")
  .option("mode", "DROPMALFORMED")
  .csv("/FileStore/tables/robusta_data_cleaned.csv")

//Combine both tables
val total = arabica.union(robusta)
val df = total.toDF(total.columns.map(_.replace(".", "_")): _*)

//Merging Null and Other
val coffee = df.na.fill("Other",Seq("Variety")).na.fill("Other",Seq("Processing_Method"))

//For convenience later
val scores = Array("Aroma","Flavor","Aftertaste","Acidity","Body","Balance","Uniformity","Clean_Cup","Sweetness","Cupper_Points")

coffee.createOrReplaceTempView("coffee")

// COMMAND ----------

//Baseline Score Statistics
val forPercentile = coffee.withColumn("always_true",lit("true")) //Need this so we can perform percentile over whole dataset
forPercentile.createOrReplaceTempView("coffee_p")
for(score <- scores) {
  coffee.select(score).agg(avg(score), stddev_pop(score)).show()
  spark.sql(s"""
  select always_true,percentile($score,0.5)
  from coffee_p
  group by always_true
  """).show()
}
coffee.select("Total_Cup_Points").agg(avg("Total_Cup_Points"), stddev_pop("Total_Cup_Points")).show()

spark.sql("""
  select always_true,percentile(Total_Cup_Points,0.5)
  from coffee_p
  group by always_true
  """).show()

// COMMAND ----------

//Average Score by Number of Submissions
display(coffee.groupBy("Owner").agg(
  "Total_Cup_Points" -> "count",
  "Total_Cup_Points" -> "mean"
))

// COMMAND ----------

//How many passed/failed
val passed = coffee.where("Total_Cup_Points>=80").count()
val notPassed = coffee.where("Total_Cup_Points<80").count()

// COMMAND ----------

//Use csv from below command with plotly script
display(coffee.select("Aroma","Flavor","Aftertaste","Acidity","Body","Balance","Uniformity","Clean_Cup","Sweetness","Cupper_Points"))

// COMMAND ----------

//Linear Regression?
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
val scoreMatrix = Array.ofDim[Double](10,10)
for(i <- 0 to 9) {
  for(j <- 0 to 9) {
    val score1 = scores(i)
    val score2 = scores(j)
    val lr = new LinearRegression()
    val va = new VectorAssembler()
      .setInputCols(Array(score1))
      .setOutputCol("features")
    val data = va.transform(coffee).select("features",score2).withColumnRenamed(score2,"label")
    val model = lr.fit(data)
    print(score1+" "+score2+" ")
    println(model.summary.rootMeanSquaredError)
    scoreMatrix(i)(j) = model.summary.rootMeanSquaredError
  }
  println()
}

// COMMAND ----------

//Print Results of Linear Regressions (Reformat into csv for excel formatting)
for(i <- 0 to 9) {
  for(j <- 0 to 9) {
    val acc = scoreMatrix(i)(j)
    print(f"'$acc%1.5f', ")
  }
  println()
}

// COMMAND ----------

//Most Popular Varieties
display(spark.sql("""
SELECT Variety,count(*)
from coffee
group by Variety
order by count(*) DESC
"""))


// COMMAND ----------

//Best Varieties for individual scores
for(score <- scores) {
  coffee.groupBy("Variety").agg(
    score -> "count",
    score -> "min",
    score -> "mean",
    score -> "max"
  ).orderBy(desc("max("+score+")")).show(false)
}

// COMMAND ----------

//Best Varieties for individual of varieties having more than 10 gradings
for(score <- scores) {
  coffee.groupBy("Variety").agg(
    score -> "count",
    score -> "mean"
  ).where("count("+score+")>10").orderBy(desc("avg("+score+")")).show(1,false)
}

// COMMAND ----------

//Best Varieties for Individual Scores
for(score <- scores) {
  coffee.groupBy("Processing_Method").agg(
    score -> "count",
    score -> "min",
    score -> "mean",
    score -> "max"
  ).orderBy(desc("avg("+score+")")).show(false)
}

// COMMAND ----------

//Most Popular Processing Methods
display(spark.sql("""
SELECT Processing_Method,count(*)
from coffee
group by Processing_Method
order by count(*) DESC
"""))

// COMMAND ----------

//Counties that submit the most coffee
display(spark.sql("""
select Country_of_Origin, count(*)
from coffee
group by Country_of_Origin
order by count(*) DESC
"""))

// COMMAND ----------

//Counties that produce the best coffee (Can change line 8 to order by mean to see)
for(score <- scores) {
  coffee.groupBy("Country_of_Origin").agg(
    score -> "count",
    score -> "min",
    score -> "mean",
    score -> "max"
  ).orderBy(desc("max("+score+")")).show(false)
}

// COMMAND ----------

//Countries that produce the best coffee on average (having more than 10 gradings)
for(score <- scores) {
  coffee.groupBy("Country_of_Origin").agg(
    score -> "count",
    score -> "mean"
  ).where("count("+score+")>10").orderBy(desc("avg("+score+")")).show(1,false)
}

// COMMAND ----------

//For Overall Summary (Variety)
coffee.groupBy("Variety").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Variety").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(asc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Variety").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("max(Total_Cup_Points)")).show(1,false)



// COMMAND ----------

//Best and Worst Processing Methods (For Overall Summary)
coffee.groupBy("Processing_Method").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Processing_Method").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(asc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Processing_Method").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("max(Total_Cup_Points)")).show(1,false)

// COMMAND ----------

//Best and worst countries for coffee (For Overall Summary)
coffee.groupBy("Country_of_Origin").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Country_of_Origin").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(asc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Country_of_Origin").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("max(Total_Cup_Points)")).show(1,false)

// COMMAND ----------

//Best and Worst Producers for coffee (For Overall Summary)
coffee.groupBy("Owner").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Owner").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(asc("avg(Total_Cup_Points)")).show(1,false)

coffee.groupBy("Owner").agg(
    "Total_Cup_Points" -> "min",
    "Total_Cup_Points" -> "mean",
    "Total_Cup_Points" -> "max"
  ).orderBy(desc("max(Total_Cup_Points)")).show(1,false)

// COMMAND ----------

//Most Prolific Producers (Top 5)
display(spark.sql("""
SELECT Owner,count(*)
from coffee
group by Owner
order by count(*) DESC
limit 5
"""))

// COMMAND ----------

//Best Producers on average having more than 10 gradings
for(score <- scores) {
  coffee.groupBy("Owner").agg(
    score -> "count",
    score -> "mean"
  ).where("count("+score+")>10").orderBy(desc("avg("+score+")")).show(1,false)
}

// COMMAND ----------

//Need to drop unusable Harvest Year Formats
val parseInt = udf((s:String) => scala.util.Try{Some(s.toInt)}.getOrElse(None))

val fixedYear = coffee.withColumn("year",parseInt($"Harvest_Year"))
val fixedNull = fixedYear.filter(fixedYear.col("year").isNotNull)
//When the most coffee was produced
display(fixedNull.groupBy("year").count().orderBy(asc("year")))


// COMMAND ----------

//Best and Worst years for coffee in certain countries
fixedNull.groupBy("Country_of_Origin","year").agg(
  "Total_Cup_Points" -> "count",
  "Total_Cup_Points" -> "average",
  "Total_Cup_Points" -> "max"
).show()

fixedNull.groupBy("Country_of_Origin","year").agg(
  "Total_Cup_Points" -> "count",
  "Total_Cup_Points" -> "average",
  "Total_Cup_Points" -> "max"
).where("count(Total_Cup_Points)>10").orderBy(desc("avg(Total_Cup_Points)")).show()

fixedNull.groupBy("Country_of_Origin","year").agg(
  "Total_Cup_Points" -> "count",
  "Total_Cup_Points" -> "average",
  "Total_Cup_Points" -> "max"
).orderBy(desc("count(Total_Cup_Points)")).show()

// COMMAND ----------

//Altitude Analysis (Exported to Excel for the graph)
val fixedAlt = coffee
  .withColumn("alt",parseInt($"altitude_mean_meters"))
  .filter($"alt"<10000)
display(fixedAlt.select("Total_Cup_Points","alt"))
