#!/bin/bash
SetupFILE=./Setup.py
libFILE=../libs
buildFILE=../buildLogs

check_and_mkdir(){
	if [ -d "$1" ];
		then
		printf "$1 found in working directory, continuing build\n"
	else
		printf "$1 not found in working directory, creating directory and continuing build\n"
		mkdir $1
	fi
}

check_and_mkSetup(){
	if [ -f "$SetupFILE" ];
		then
		printf "$SetupFILE exists in working directory.\n Compiling $1 ...\n"
	else
		printf "$SetupFILE does not exist.\n Creating setup file and Compiling $1 ...\n"
		touch $SetupFILE
		echo "from distutils.core import setup" >> $SetupFILE
		echo "from Cython.Build import cythonize" >> $SetupFILE
		echo "setup(ext_modules = cythonize" >> $SetupFILE
	fi
}

check_and_mkdir $libFILE
check_and_mkdir $buildFILE
check_and_mkSetup

sed -i "s;^setup(ext_modules.*;setup(ext_modules = cythonize('$1', compiler_directives = {'language_level' : '3'}));" $SetupFILE
python3 Setup.py build_ext --inplace > ./logBuild.txt
mv *.so $libFILE
mv *.c $libFILE
rnd="$RANDOM"
mv build "build_$rnd"
mv logBuild.txt "logBuild_$rnd.txt"
mv -t $buildFILE "build_$rnd" "logBuild_$rnd.txt"
printf "Shared Object (.so) and C-file (.c) created and placed inside $libFILE\n"
