#!/bin/sh 
#
convertCodeFilePath=$1 
fromCode=$2 
toCode=$3 
  
for i in {1..1} 
do
  [ -f $convertCodeFilePath ] 
  if [ $? -eq 0 ] 
  then
    iconv -f $fromCode -t $toCode -c -o $convertCodeFilePath $convertCodeFilePath 
    if [ $? -ne 0 ] 
    then
      echo $convertCodeFilePath "=>" convert code failed.      
    else
      echo $convertCodeFilePath "=>" convert code success. 
    fi
    break; 
  fi

  [ -d $convertCodeFilePath ] 
  if [ $? -ne 0 ] 
  then
    break; 
  fi
      
  dir=`ls $convertCodeFilePath | sort -d` 

  for fileName in $dir
  do
    fileFullPatch=$convertCodeFilePath/$fileName 

    fileType=`echo $fileName |awk -F. '{print $2}'` 

    [ -d $fileName ] 
    if [ $? -eq 0 ] 
    then
      continue
    fi

    if [ $fileType != 'sh' ] && [ $fileType != 'py' ] && [ $fileType != 'xml' ] && [ $fileType != 'properties' ] && \
       [ $fileType != 'q' ] && [ $fileType != 'hql' ] && [ $fileType != 'txt' ] 
    then
      continue
    fi

    iconv -f $fromCode -t $toCode -c -o $fileFullPatch $fileFullPatch 
    if [ $? -ne 0 ] 
    then
      echo $fileName "=>" convert code failed. 
      continue
    else
      echo $fileName "=>" convert code success. 
    fi
  done
done