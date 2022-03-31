#!/bin/bash
source envars.sh

#curl -D- -X POST $DB_URL --data-urlencode "update=drop all"
curl -D- -H "Content-Type: application/xml" --upload-file IEEE123Sec.xml -X POST $DB_URL

