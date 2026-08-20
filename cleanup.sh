#! /bin/bash
rm -rf

echo "Removing temporary files..."

if [ -d "./temp" ]; then
    rm -f ./temp/clean_*.csv
    rm -f ./temp/glitch_*.csv
    rm -f ./temp/splits_*.txt
fi

echo "Temporary files removed"