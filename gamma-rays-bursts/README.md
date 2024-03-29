# Gamma Ray Bursts Clustering

Here, I estimate the number of unique categories of GRBs using clustering algorithms. The number of unique categories (classes) is unknown to the algorithm (and to me). The following algorithms are used:
* Fuzzy C-Means
* K-Means
* MBSAS

## Background
Gamma ray bursts (GRB) are highly energetic sources of gamma radiation that are observed in the universe.  Historically, they were originally discovered by gamma ray detectors on board the VELA satellites which were originally designed to search for nuclear detonations in space.  These events were characterized by extremely energetic bursts of gamma rays which lasted from milliseconds to seconds in duration.  Later, in 1978, gamma ray detectors on-board the Pioneer –Venus Orbiter also detected these bursts of gamma rays and measured their properties.  Since the detectors on PVO were omni-directional, it was unclear whether these bursts were coming from within our own galaxy, or from the farthest reaches of the universe.  It was reasoned that if one could identify different classes or categories of GRB, then perhaps each of these classes might be attributed to different distributions or origins of the bursts.  Since there is no ground truth or training data available, an unsupervised analysis is required  to attempt to categorize GRBs.

## Dataset:
The available data consist of the features extracted from 99 GRBs that were measured by the PVO sensors.  These are provided in the GRBData.xlsx spreadsheet.  There is one data line for each exemplar (GRB).  The feature data on each line is structured as follows:

Column # | GRB Features
--- | ---
   1	|	year of event
   2	|	month of event
   3	|	day of event
   4	|	seconds of the day of the event
   5	|	Log (Burst Duration) - in seconds
   6	|	Duration Flag (0 or 1)
   7	|	Average Time(sec)
   8	|	Second Moment (sec)
   9	|	Third Moment (sec)
   10	|	log (Total Energy)
   11	|	log (Early Energy/Late Energy)
   12	|	Total High Energy 
   13	|	Early High Energy 
   14	|	Late High Energy	
   15	|	log (Peak Energy Rate)	
   16	|	Peak Energy Rate Time (sec)	
   17	|	log (Peak Count Rate at Peak)	
   18	|	Time of Peak Count Rate (sec)	
   19	|	Number of Peaks in Signal	
   20	|	Signal Run Length	
   21	|	Signal Fractal Dimension	
   22	|	Slope of the Signal Cepstrum	
   23	|	Wavelet Transform (res 5) - 0 Crossings	
   24	|	Wavelet Transform (res 6) - 0 Crossings	
   25	|	Wavelet Transform (res 7) - 0 Crossings	
   26	|	Wavelet Transform (res 8) - 0 Crossings	
   27	|	Fractal Dimension of Smoothed Signal	
   28	|	Number of -3db Peaks (Smoothed Signal)	
   29	|	Separation of Two Largest Peaks	
   30	|	Ratio of Two Largest Peaks
