wget http://rpg.ifi.uzh.ch/datasets/gehrig_et_al_iccv19/N-Caltech101.zip
unzip N-Caltech101.zip -d N-Caltech101-RPG
cd N-Caltech101-RPG
ln -s training train
ln -s testing test
cd ..
rm N-Caltech101.zip