mkdir download; cd download
mkdir outdoor_day; cd outdoor_day

echo "Downloading outdoor_day1 data..."
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_data.bag

echo "Downloading outdoor_day2 data..."
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day2_data.bag


cd ..
mkdir indoor_flying; cd indoor_flying

echo "Downloading indoor_flying1 data..."
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying1_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying1_gt.bag

echo "Downloading indoor_flying1 data..."
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying2_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying2_gt.bag

echo "Downloading indoor_flying1 data..."
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying3_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying3_gt.bag
