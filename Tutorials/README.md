This folder is created to hold tutorials for working with various parts of the Reco and Analysis code. 

## make_event_display:
  * <b> Input: </b> any version of dlreco/merged dlana file with wire image and Pgraph tree of reco vertices
  * <b> Output: </b> png event display for each plane of each entry
  * <b> Simple Usage: </b> ```./make_event_display <input filename> <output dir of pngs>```
  * <b> contents: </b> root 2d histgrams (filling and drawing), loading analysis files, basic usage of larcv IO Manager

## create_larcvImage2D:
  * <b> Input: </b> any version of dlreco/merged dlana file with wire larcv Image2d
  * <b> Output: </b> a root file containing a copy of the Image2d tree. (now in tick forward direction)
  * <b> Simple Usage: </b> ```./create_larcvImage2d <input filename> <output dir root file>```
  * <b> What the tutorial looks at: </b>  loading analysis files, basic usage of larcv IO Manager, reading and writing larcv::Image2D

## GetTruthInfo:
 * <b> Input: </b> any version of dlreco/merged dlana file with larlite truth info
 * <b> Output: cout statements of truth info for each event
 * <b> Simple Usage: </b> ```./GetTruthInfo <input filename>```
 * <b> What the tutorial looks at: </b>  loading analysis files, basic usage of larlite IO Manager, working with mc truth objects
  
