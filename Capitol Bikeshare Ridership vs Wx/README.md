# To bike or not to bike? Correlations between Capital Bikeshare ridership and weather events.

In this analysis, we explore and explain the effect of weather events in the D.C. Metropolitan area on the number of bike rentals. 

Also, several models are built using linear regression. Confidence intervals are constructed by bootstraping. (i.e. resampling with substitution). These models attempt to predict ridership based on several factors studied.  

Last but not least, this effort was possible thanks to the open data stance at Capital Bikeshare and WeatherUnderground.com. :)

--------------------------------------------------------------------------------------------------------------------------

There are two phases:
    a) EDA phase of project (Exploratory Data Analysis)
    b) "Infographic" phase

To reproduce our EDA, follow these steps:

1. Ensure you have all 2012 and 2013 csv files found at 
https://www.capitalbikeshare.com/trip-history-data

2. Run Project_EDA_BikeDataReader.ipynb to process the Capital Bikeshare files

3. Run Project_EDA_wxDataReaderHourlyData.ipynb to get the weather data

To run the analysis on a Jupiter notebook:
4. Run Project_EDA_Main.ipynb

An infographic with a summary of the analysis can be seen in infographic.pdf
5. The raw plots used for the infographic are found in InfographicRawPlots.ipynb

--------------------------------------------------------------------------------------------------------------------------
The exact files we used in our EDA can be found at:

wxData.tsv= https://drive.google.com/file/d/0B2nonbXWDbpDZkdZaWJvTjZsWGc/view?usp=sharing

master.tsv = https://drive.google.com/file/d/0B2nonbXWDbpDUUZiV2hIVTNqd0k/view?usp=sharing

wxData folder (has individual files) : https://drive.google.com/folderview?id=0B2nonbXWDbpDY3RFRHhEZEhqazQ&usp=sharing

bikeData folder (has quarterly data from Capital Bikeshare): 
https://drive.google.com/folderview?id=0B2nonbXWDbpDRGZvTzIwRTJJV0k&usp=sharing


![Infographic](https://github.com/ajRiverav/jhuProjects/blob/master/Capitol%20Bikeshare%20Ridership%20vs%20Wx/Infographic.jpg?raw=true)


