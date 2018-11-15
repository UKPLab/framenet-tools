#Downloading required files
if [ -d "scripts" ]; then
	echo "[Skip] Already found scripts!"
else
	echo "Downloading scripts!"
	wget https://github.com/akb89/pyfn/releases/download/v1.0.0/scripts.7z
	7z x scripts.7z
	rm scripts.7z
fi

if [ -d "lib" ]; then
	echo "[Skip] Already found lib!"
else
	echo "Downloading lib!"
	wget https://github.com/akb89/pyfn/releases/download/v1.0.0/lib.7z
	7z x lib.7z
	rm lib.7z
fi

if [ -d "resources" ]; then
	echo "[Skip] Already found resources!"
else
	echo "Downloading resources!"
	wget https://github.com/akb89/pyfn/releases/download/v1.0.0/resources.7z
	7z x resources.7z
	rm resources.7z
fi

#First, convert the data to CoNLL format
pyfn convert \
  --from fnxml \
  --to semafor \
  --source data/fndata-1.5-with-dev \
  --target data/experiments/xp_001/data \
  --splits train \
  --output_sentences \

#Then append it with POS-tags and dependency information
chmod -R +x scripts
cd scripts/
./preprocess.sh -x 001 -t nlp4j -d bmst -p semafor
