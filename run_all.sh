#!/bin/bash

for assignment in 'Crises' 'Domicile' 'CAT' 'CMS' 'Prestataires' 'RTC' 'Nuit' 'Médical' 'Services' 'SAP' 'RENAULT' 'Gestion'  'Téléphonie' 'Manager' 'Japon' 'Mécanicien' 
do
	echo $assignment
	python ./code/pipeline.py $assignment $1
done

echo "Tech.\ Inter"
python ./code/pipeline.py Tech.\ Inter $1
echo "Tech.\ Total"
python ./code/pipeline.py Tech.\ Total $1
echo "Gestion\ Relation\ Clienteles"
python ./code/pipeline.py Gestion\ Relation\ Clienteles $1
echo "Tech.\ Axa "
python ./code/pipeline.py Tech.\ Axa $1
echo "Gestion\ Assurances"
python ./code/pipeline.py Gestion\ Assurances $1
echo "Gestion\ Clients"
python ./code/pipeline.py Gestion\ Clients $1
echo "Gestion\ Renault"
python ./code/pipeline.py Gestion\ Renault $1
echo "Gestion\ \-\ Accueil\ Telephonique"
python ./code/pipeline.py Gestion\ \-\ Accueil\ Telephonique $1
echo "Gestion\ DZ"
python ./code/pipeline.py Gestion\ DZ $1
echo "Regulation\ Medicale"
python ./code/pipeline.py Regulation\ Medicale $1
