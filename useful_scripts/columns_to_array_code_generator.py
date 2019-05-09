databaseName = 'ahriWithGeoArray'
newDatabaseName = 'ahriWithGeoArray'
newName = 'diastolicBloodPressure'
numColumns = 3
secondDimension = True
numColumnsSecondDimension = 2

#CONVERT COLUMNS INTO ARRAY
result = "db." + databaseName + ".aggregate( [{ $addFields: { '" + newName + "': ["

if secondDimension:
	for x in range(0, numColumns - 1):
		result += "["
		for y in range(0, numColumnsSecondDimension - 1):
			tmp =  "'$" + newName + str(x) + str(y) + "',"
			result += tmp
		lastLine2nd = "'$" + newName + str(x) + str(numColumnsSecondDimension - 1) + "'], "
		result += lastLine2nd
	result += "["
	for y in range(0, numColumnsSecondDimension - 1):
		tmp =  "'$" + newName + str(numColumns - 1) + str(y) + "',"
		result += tmp
	lastLine = "'$" + newName + str(numColumns - 1) + str(numColumnsSecondDimension - 1) + "']"
	result += lastLine
else:
	for x in range(0, numColumns - 1):
		tmp =  "'$" + newName + str(x) + "',"
		result += tmp
	lastLine = "'$" + newName + str(numColumns - 1) + "'"
	result += lastLine

tmp = "] } },{ '$out': '" + newDatabaseName + "' }])"
result += tmp

print("\nCONVERT COLUMNS INTO ARRAY:")
print(result)


#DELETE FIELDS FROM COLLECTION
# in case dataBaseName and newDatabaseName are the same

result = "db." + newDatabaseName + ".updateMany({}, {$unset: {"

if secondDimension:
	for x in range(0, numColumns - 1):
		for y in range(0, numColumnsSecondDimension):
			tmp =  "'" + newName + str(x) + str(y) + "': 1,"
			result += tmp
	for y in range(0, numColumnsSecondDimension - 1):
			tmp =  "'" + newName + str(numColumns - 1) + str(y) + "': 1,"
			result += tmp
	lastLine = "'" + newName + str(numColumns - 1) + str(numColumnsSecondDimension - 1) + "': 1"
	result += lastLine
else:
	for x in range(0, numColumns - 1):
		tmp =  "'" + newName + str(x) + "': 1,"
		result += tmp
	lastLine = "'" + newName + str(numColumns - 1) + "': 1"
	result += lastLine

result += "}})"

print("\nDELETE FIELDS FROM COLLECTION:")
print(result)

"""
# ADD NEW FIELD FROM OLD FIELDS
result = "db." + databaseName + ".aggregate( [{ $addFields: { '" + newName + "': { $concat: ["

for x in range(0, numColumns - 1):
	tmp =  "'$" + newName + str(x) + "', ' ; ',"
	result += tmp
lastLine = "'$" + newName + str(numColumns - 1) + "'"
result += lastLine

tmp = "] } } },{ '$out': '" + newDatabaseName + "' }])"
result += tmp

print("ADD NEW FIELD FROM OLD FIELDS:")
print(result)


#SWITCH FROM KOMMA SEPERATED STRING TO ARRAY OF STRINGS

result = "db." + newDatabaseName + ".aggregate([{ '$addFields': { '" + newName + "': { '$cond': [{ '$eq': [{ '$type': '" + newName + "' }, 'string']},{ '$split': [ '$" + newName + "', ' ; ' ]}, '$" + newName + "']}}},{ '$out': '" + newDatabaseName + "' }])"

print("\nSWITCH FROM KOMMA SEPERATED STRING TO ARRAY OF STRINGS:")
print(result)
"""