import sddsdata, sys, time

class SDDS:
     """This class implements SDDS datasets."""

     #define common SDDS definitions
     SDDS_VERBOSE_PrintErrors = 1
     SDDS_EXIT_PrintErrors = 2
     SDDS_CHECK_OKAY = 0
     SDDS_CHECK_NONEXISTENT = 1
     SDDS_CHECK_WRONGTYPE = 2
     SDDS_CHECK_WRONGUNITS = 3
     SDDS_LONGDOUBLE = 1
     SDDS_DOUBLE = 2
     SDDS_REAL64 = 2
     SDDS_FLOAT = 3
     SDDS_REAL32 = 3
     SDDS_LONG64 = 4
     SDDS_INT64 = 4
     SDDS_ULONG64 = 5
     SDDS_UINT64 = 5
     SDDS_LONG = 6
     SDDS_INT32 = 6
     SDDS_ULONG = 7
     SDDS_UINT32 = 7
     SDDS_SHORT = 8
     SDDS_INT16 = 8
     SDDS_USHORT = 9
     SDDS_UINT16 = 9
     SDDS_STRING = 10
     SDDS_CHARACTER = 11
     SDDS_NUM_TYPES = 11
     SDDS_BINARY = 1
     SDDS_ASCII = 2
     SDDS_FLUSH_TABLE = 1

     def __init__(self, index):
          #only indexes of 0 through 19 are allowed
          if index >= 0 and index < 20:
               self.index = index
          else:
               self.index = 0
          #initialize data storage variables
          self.description = ["", ""]
          self.parameterName = []
          self.arrayName = []
          self.columnName = []
          self.parameterDefinition = []
          self.arrayDefinition = []
          self.arrayDimensions = []
          self.columnDefinition = []
          self.parameterData = []
          self.arrayData = []
          self.columnData = []
          self.mode = self.SDDS_ASCII
     
     def load(self, input):
          """Load an SDDS file into an SDDS class."""
          
          try:
               #open SDDS file
               if sddsdata.InitializeInput(self.index, input) != 1:
                    time.sleep(1)
                    if sddsdata.InitializeInput(self.index, input) != 1:
                         raise ValueError

               #get data storage mode (SDDS_ASCII or SDDS_BINARY)
               self.mode = sddsdata.GetMode(self.index)

               #get description text and contents
               self.description = sddsdata.GetDescription(self.index)

               #get parameter names
               self.parameterName = sddsdata.GetParameterNames(self.index)
               numberOfParameters = len(self.parameterName)

               #get array names
               self.arrayName = sddsdata.GetArrayNames(self.index)
               numberOfArrays = len(self.arrayName)

               #get column names
               self.columnName = sddsdata.GetColumnNames(self.index)
               numberOfColumns = len(self.columnName)

               #get parameter definitions
               self.parameterDefinition = list(range(numberOfParameters))
               for i in range(numberOfParameters):
                    self.parameterDefinition[i] = sddsdata.GetParameterDefinition(self.index, self.parameterName[i])

               #get array definitions
               self.arrayDefinition = list(range(numberOfArrays))
               for i in range(numberOfArrays):
                    self.arrayDefinition[i] = sddsdata.GetArrayDefinition(self.index, self.arrayName[i])

               #get column definitions
               self.columnDefinition = list(range(numberOfColumns))
               for i in range(numberOfColumns):
                    self.columnDefinition[i] = sddsdata.GetColumnDefinition(self.index, self.columnName[i])

               #initialize parameter, array and column data
               self.parameterData = list(range(numberOfParameters))
               self.arrayData = list(range(numberOfArrays))
               self.columnData = list(range(numberOfColumns))
               for i in range(numberOfParameters):
                    self.parameterData[i] = []
               for i in range(numberOfArrays):
                    self.arrayData[i] = []
               for i in range(numberOfColumns):
                    self.columnData[i] = []
                    
               self.arrayDimensions = list(range(numberOfArrays))
               for i in range(numberOfArrays):
                    self.arrayDimensions[i] = []

               #read in SDDS data
               page = sddsdata.ReadPage(self.index)
               if page != 1:
                    raise Exception("Unable to read SDDS data for the first page")
               while page > 0:
                    for i in range(numberOfParameters):
                         self.parameterData[i].append(sddsdata.GetParameter(self.index,i))
                    for i in range(numberOfArrays):
                         self.arrayData[i].append(sddsdata.GetArray(self.index,i))
                         self.arrayDimensions[i].append(sddsdata.GetArrayDimensions(self.index,i))
                    rows = sddsdata.RowCount(self.index);
                    if rows > 0:
                         for i in range(numberOfColumns):
                              self.columnData[i].append(sddsdata.GetColumn(self.index,i))
                    else:
                         for i in range(numberOfColumns):
                              self.columnData[i].append([])
                    
                    page = sddsdata.ReadPage(self.index)

               #close SDDS file
               if sddsdata.Terminate(self.index) != 1:
                    raise ValueError
               
          except:
               sddsdata.PrintErrors(self.SDDS_VERBOSE_PrintErrors)
               raise

     def save(self, output):
          """Save an SDDS class to an SDDS file."""

          try:
               #check for invalid SDDS data
               numberOfParameters = len(self.parameterName)
               numberOfArrays = len(self.arrayName)
               numberOfColumns = len(self.columnName)
               pages = 0
               if numberOfParameters != len(self.parameterData):
                    raise Exception("unmatched parameterName and parameterData")
               if numberOfArrays != len(self.arrayData):
                    raise Exception("unmatched arrayName and arrayData")
               if numberOfColumns != len(self.columnData):
                    raise Exception("unmatched columnName and columnData")
               if numberOfParameters != len(self.parameterDefinition):
                    raise Exception("unmatched parameterName and parameterDefinition")
               if numberOfArrays != len(self.arrayDefinition):
                    raise Exception("unmatched arrayName and arrayDefinition")
               if numberOfColumns != len(self.columnDefinition):
                    raise Exception("unmatched columnName and columnDefinition")
               if numberOfParameters > 0:
                    pages = len(self.parameterData[0])
               elif numberOfColumns > 0:
                    pages = len(self.columnData[0])
               elif numberOfArrays > 0:
                    pages = len(self.arrayData[0])
               for i in range(numberOfParameters):
                    if pages != len(self.parameterData[i]):
                         raise Exception("unequal number of pages in parameter data")
               for i in range(numberOfArrays):
                    if pages != len(self.arrayData[i]):
                         raise Exception("unequal number of pages in array data")
                    if pages != len(self.arrayDimensions[i]):
                         raise Exception("unequal number of pages in array dimension data")
               for i in range(numberOfColumns):
                    if pages != len(self.columnData[i]):
                         raise Exception("unequal number of pages in column data")
               for page in range(pages):               
                    rows = 0
                    if numberOfColumns > 0:
                         rows = len(self.columnData[0][page])
                    for i in range(numberOfColumns):
                         if rows != len(self.columnData[i][page]):
                              raise Exception("unequal number of rows in column data")

               #open SDDS output file
               if sddsdata.InitializeOutput(self.index, self.mode, 1, self.description[0], self.description[1], output) != 1:
                    raise ValueError

               #define parameters, arrays and columns
               for i in range(numberOfParameters):
                    if sddsdata.DefineParameter(self.index, self.parameterName[i],
                                            self.parameterDefinition[i][0],
                                            self.parameterDefinition[i][1],
                                            self.parameterDefinition[i][2],
                                            self.parameterDefinition[i][3],
                                            self.parameterDefinition[i][4],
                                            self.parameterDefinition[i][5]) == -1:
                         raise ValueError
               for i in range(numberOfArrays):
                    if sddsdata.DefineArray(self.index, self.arrayName[i],
                                            self.arrayDefinition[i][0],
                                            self.arrayDefinition[i][1],
                                            self.arrayDefinition[i][2],
                                            self.arrayDefinition[i][3],
                                            self.arrayDefinition[i][4],
                                            self.arrayDefinition[i][5],
                                            self.arrayDefinition[i][6],
                                            self.arrayDefinition[i][7]) == -1:
                         raise ValueError
               for i in range(numberOfColumns):               
                    if sddsdata.DefineColumn(self.index, self.columnName[i],
                                            self.columnDefinition[i][0],
                                            self.columnDefinition[i][1],
                                            self.columnDefinition[i][2],
                                            self.columnDefinition[i][3],
                                            self.columnDefinition[i][4],
                                            self.columnDefinition[i][5]) == -1:
                         raise ValueError

               #write SDDS header
               if sddsdata.WriteLayout(self.index) != 1:
                    raise ValueError

               #write SDDS data
               for page in range(pages):               
                    rows = 0
                    if numberOfColumns > 0:
                         rows = len(self.columnData[0][page])
                    if sddsdata.StartPage(self.index, rows) != 1:
                         raise ValueError
                    for i in range(numberOfParameters):
                         if sddsdata.SetParameter(self.index, i, self.parameterData[i][page]) != 1:
                              raise ValueError
                    for i in range(numberOfArrays):
                         if sddsdata.SetArray(self.index, i, self.arrayData[i][page], self.arrayDimensions[i][page]) != 1:
                              raise ValueError
                    for i in range(numberOfColumns):
                         if sddsdata.SetColumn(self.index, i, self.columnData[i][page]) != 1:
                              raise ValueError
                    if sddsdata.WritePage(self.index) != 1:
                         raise ValueError

               #close SDDS output file
               if sddsdata.Terminate(self.index) != 1:
                    raise ValueError
          except:
               sddsdata.PrintErrors(self.SDDS_VERBOSE_PrintErrors)
               raise


     def setDescription(self, text, contents):
          self.description = [text, contents]

     def defineParameter(self, name, symbol, units, description, formatString, type, fixedValue):
          self.parameterName.append(name)
          self.parameterDefinition.append([symbol, units, description, formatString, type, fixedValue])
          self.parameterData.append([])
          
     def defineSimpleParameter(self, name, type):
          self.parameterName.append(name)
          self.parameterDefinition.append(["", "", "", "", type, ""])
          self.parameterData.append([])
          
     def defineArray(self, name, symbol, units, description, formatString, group_name, type, fieldLength, dimensions):
          self.arrayName.append(name)
          self.arrayDefinition.append([symbol, units, description, formatString, group_name, type, fieldLength, dimensions])
          self.arrayData.append([])
          self.arrayDimensions.append([])
          
     def defineSimpleArray(self, name, type, dimensions):
          self.arrayName.append(name)
          self.arrayDefinition.append(["", "", "", "", "", type, 0, dimensions])
          self.arrayData.append([])
          self.arrayDimensions.append([])

     def defineColumn(self, name, symbol, units, description, formatString, type, fieldLength):
          self.columnName.append(name)
          self.columnDefinition.append([symbol, units, description, formatString, type, fieldLength])
          self.columnData.append([])
          
     def defineSimpleColumn(self, name, type):
          self.columnName.append(name)
          self.columnDefinition.append(["", "", "", "", type, 0])
          self.columnData.append([])

     def setParameterValueList(self, name, valueList):
          numberOfParameters = len(self.parameterName)
          for i in range(numberOfParameters):
               if self.parameterName[i] == name:
                    self.parameterData[i] = valueList
                    return
          msg = "invalid parameter name " + name
          raise Exception(msg)
     
     def setParameterValue(self, name, value, page):
          page = page - 1
          numberOfParameters = len(self.parameterName)
          for i in range(numberOfParameters):
               if self.parameterName[i] == name:
                    if len(self.parameterData[i]) == page:
                         self.parameterData[i][page:] = [value]
                    elif len(self.parameterData[i]) < page or page < 0:
                         msg = "invalid page " + str(page+1)
                         raise Exception(msg)
                    else:
                         self.parameterData[i][page] = [value]
                    return
          msg = "invalid parameter name " + name
          raise Exception(msg)

     def setArrayValueLists(self, name, valueList, dimensionList):
          numberOfArrays = len(self.arrayName)
          for i in range(numberOfArrays):
               if self.arrayName[i] == name:
                    self.arrayDimensions[i] = dimensionList
                    self.arrayData[i] = valueList
                    return
          msg = "invalid array name " + name
          raise Exception(msg)
     
     def setArrayValueList(self, name, valueList, dimensionList, page):
          page = page - 1
          numberOfArray = len(self.arrayName)
          for i in range(numberOfArrays):
               if self.arrayName[i] == name:
                    if len(self.arrayData[i]) == page:
                         self.arrayData[i][page:] = [valueList]
                         self.arrayDimensions[i][page:] = [dimensionList]
                    elif len(self.arrayData[i]) < page or page < 0:
                         msg = "invalid page " + str(page+1)
                         raise Exception(msg)
                    else:
                         self.arrayData[i][page] = [valueList]
                         self.arrayDimensions[i][page] = [dimensionList]
                    return
          msg = "invalid array name " + name
          raise Exception(msg)

     def setColumnValueLists(self, name, valueList):
          numberOfColumns = len(self.columnName)
          for i in range(numberOfColumns):
               if self.columnName[i] == name:
                    self.columnData[i] = valueList
                    return
          msg = "invalid column name " + name
          raise Exception(msg)
     
     def setColumnValueList(self, name, valueList, page):
          page = page - 1
          numberOfColumns = len(self.columnName)
          for i in range(numberOfColumns):
               if self.columnName[i] == name:
                    if len(self.columnData[i]) == page:
                         self.columnData[i][page:] = [valueList]
                    elif len(self.columnData[i]) < page or page < 0:
                         msg = "invalid page " + str(page+1)
                         raise Exception(msg)
                    else:
                         self.columnData[i][page] = [valueList]
                    return
          msg = "invalid column name " + name
          raise Exception(msg)
     
     def setColumnValue(self, name, value, page, row):
          page = page - 1
          row = row - 1
          numberOfColumns = len(self.columnName)
          for i in range(numberOfColumns):
               if self.columnName[i] == name:
                    if len(self.columnData[i]) == page:
                         if row == 0:
                              self.columnData[i][page:] = [[value]]
                         else:
                              msg = "invalid row " + str(row+1)
                              raise Exception(msg)
                    elif len(self.columnData[i]) < page or page < 0:
                         msg = "invalid page " + str(page+1)
                         raise Exception(msg)
                    else:
                         if len(self.columnData[i][page]) == row:
                              self.columnData[i][page][row:] = [value]
                         elif len(self.columnData[i][page]) < row or row < 0:
                              msg = "invalid row " + str(row+1)
                              raise Exception(msg)
                         else:
                              self.columnData[i][page][row] = [value]
                    return
          msg = "invalid column name " + name
          raise Exception(msg)
     

def demo1(output):
     """Save an demo SDDS file using the SDDS class."""

     x = SDDS(0)
     x.description[0] = "text"
     x.description[1] = "contents"
     x.parameterName = ["ShortP", "LongP", "FloatP", "DoubleP", "StringP", "CharacterP"]
     x.parameterData = [[1, 6], [2, 7], [3.3, 8.8], [4.4, 9.8], ["five", "ten"], ["a", "b"]]
     x.parameterDefinition = [["","","","",x.SDDS_SHORT,""],
                                 ["","","","",x.SDDS_LONG,""],
                                 ["","","","",x.SDDS_FLOAT,""],
                                 ["","","","",x.SDDS_DOUBLE,""],
                                 ["","","","",x.SDDS_STRING,""],
                                 ["","","","",x.SDDS_CHARACTER, ""]]
     
     x.arrayName = ["ShortA", "LongA", "FloatA", "DoubleA", "StringA", "CharacterA"]
     x.arrayDefinition = [["","","","","",x.SDDS_SHORT,0,1],
                          ["","","","","",x.SDDS_LONG,0,1],
                          ["","","","","",x.SDDS_FLOAT,0,2],
                          ["","","","","",x.SDDS_DOUBLE,0,1],
                          ["","","","","",x.SDDS_STRING,0,1],
                          ["","","","","",x.SDDS_CHARACTER,0,1]]
     x.arrayDimensions = [[[6],[8]],
                          [[5],[7]],
                          [[2, 3],[2, 4]],
                          [[4],[5]],
                          [[4],[5]],
                          [[4],[5]]]
     x.arrayData = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]],
                    [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]],
                    [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]],
                    [[1, 2, 3, 4], [1, 2, 3, 4, 5]],
                    [["one", "two", "three", "four"], ["five", "six", "seven", "eight", "nine"]],
                    [["a", "b", "c", "d"], ["e", "f", "g", "h", "i"]]]
     
     x.columnName = ["ShortC", "LongC", "FloatC", "DoubleC", "StringC", "CharacterC"]
     x.columnData = [[[1, 2, 3], [-1, -2, -3.6, -4.4]],
                        [[1, 2, 3], [-1, -2, -3.6, -4.4]],
                        [[1, 2, 3], [-1, -2, -3.6, -4.4]],
                        [[1, 2, 3], [-1, -2, -3.6, -4.4]],
                        [["row 1", "row 2", "row 3"], ["row 1", "row 2", "row 3", "row 4"]],
                        [["x", "y", "z"], ["i", "j", "k", "l"]]]
     x.columnDefinition = [["","","","",x.SDDS_SHORT,0],
                              ["","","","",x.SDDS_LONG,0],
                              ["","","","",x.SDDS_FLOAT,0],
                              ["","","","",x.SDDS_DOUBLE,0],
                              ["","","","",x.SDDS_STRING,0],
                              ["","","","",x.SDDS_CHARACTER,0]]
     
     x.save(output)
     del x

def demo2(output):
     """Save an demo SDDS file using the SDDS class."""

     x = SDDS(0)
     x.setDescription("text", "contents")
     names = ["Short", "Long", "Float", "Double", "String", "Character"]
     types = [x.SDDS_SHORT, x.SDDS_LONG, x.SDDS_FLOAT, x.SDDS_DOUBLE, x.SDDS_STRING, x.SDDS_CHARACTER]
     for i in range(6):
          x.defineSimpleParameter(names[i] + "P", types[i])
          if types[i] == x.SDDS_FLOAT:
              x.defineSimpleArray(names[i] + "A", types[i], 2)
          else:
              x.defineSimpleArray(names[i] + "A", types[i], 1)
          x.defineSimpleColumn(names[i] + "C", types[i])
     parameterData = [[1, 6], [2, 7], [3.3, 8.8], [4.4, 9.8], ["five", "ten"], ["a", "b"]]
     for i in range(6):
          x.setParameterValueList(names[i] + "P", parameterData[i])

     arrayDimensions = [[[6],[8]],
                        [[5],[7]],
                        [[2, 3],[2, 4]],
                        [[4],[5]],
                        [[4],[5]],
                        [[4],[5]]]
     arrayData = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]],
                    [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]],
                    [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]],
                    [[1, 2, 3, 4], [1, 2, 3, 4, 5]],
                    [["one", "two", "three", "four"], ["five", "six", "seven", "eight", "nine"]],
                    [["a", "b", "c", "d"], ["e", "f", "g", "h", "i"]]]
     for i in range(6):          
          x.setArrayValueLists(names[i] + "A", arrayData[i], arrayDimensions[i])
     
     columnData = [[[1, 2, 3], [-1, -2, -3.6, -4.4]],
                   [[1, 2, 3], [-1, -2, -3.6, -4.4]],
                   [[1, 2, 3], [-1, -2, -3.6, -4.4]],
                   [[1, 2, 3], [-1, -2, -3.6, -4.4]],
                   [["row 1", "row 2", "row 3"], ["row 1", "row 2", "row 3", "row 4"]],
                   [["x", "y", "z"], ["i", "j", "k", "l"]]]
     for i in range(6):          
          x.setColumnValueLists(names[i] + "C", columnData[i])
     x.save(output)

def demo3(output):
     """Save an demo SDDS file using sddsdata commands."""

     x = SDDS(0)

     try:
         #open SDDS output file
         if sddsdata.InitializeOutput(x.index, x.SDDS_BINARY, 1, "", "", output) != 1:
              raise ValueError
         #Setting column_major to true. Only use this if you are going to write whole columns and not one row at a time.
         sddsdata.SetColumnMajorOrder(x.index)
         #define parameters
         if sddsdata.DefineSimpleParameter(x.index, "ParameterA", "mm", x.SDDS_DOUBLE) != 1:
              raise ValueError
         #define arrays, first is 1 dimensional, second is 2 dimensional
         if sddsdata.DefineSimpleArray(x.index, "ArrayA", "DegC", x.SDDS_DOUBLE, 1) != 1:
              raise ValueError
         if sddsdata.DefineSimpleArray(x.index, "ArrayB", "DegC", x.SDDS_DOUBLE, 2) != 1:
              raise ValueError
         #define columns
         if sddsdata.DefineSimpleColumn(x.index, "ColumnA", "Volts", x.SDDS_DOUBLE) != 1:
              raise ValueError
         if sddsdata.DefineSimpleColumn(x.index, "ColumnB", "Amps", x.SDDS_DOUBLE) != 1:
              raise ValueError
         #write SDDS header
         if sddsdata.WriteLayout(x.index) != 1:
              raise ValueError
         #start SDDS page. Allocate 100 rows.
         if sddsdata.StartPage(x.index, 100) != 1:
              raise ValueError
         #set parameter values
         if sddsdata.SetParameter(x.index, "ParameterA", 1.1) != 1:
              raise ValueError
         #set array values
         if sddsdata.SetArray(x.index, "ArrayA", [1, 2, 3], [3]) != 1:
              raise ValueError
         if sddsdata.SetArray(x.index, "ArrayB", [1, 2, 3, 4, 5, 6], [2, 3]) != 1:
              raise ValueError
         #set column values
         if sddsdata.SetColumn(x.index, "ColumnA", [1, 2, 3]) != 1:
              raise ValueError
         if sddsdata.SetColumn(x.index, "ColumnB", [1, 2, 3]) != 1:
              raise ValueError
         #write page to disk
         if sddsdata.WritePage(x.index) != 1:
              raise ValueError
         #we could start another page here if we wanted to
         
         #close SDDS output file
         if sddsdata.Terminate(x.index) != 1:
              raise ValueError

     except:
          sddsdata.PrintErrors(x.SDDS_VERBOSE_PrintErrors)
          raise
     
def demo4(output):
     """Save an demo SDDS file using sddsdata command and writing one row at a time."""

     x = SDDS(0)

     try:
         #open SDDS output file
         if sddsdata.InitializeOutput(x.index, x.SDDS_BINARY, 1, "", "", output) != 1:
              raise ValueError
         #turning on fsync mdoe and fixed rows count mode. These are useful for loggers
         sddsdata.EnableFSync(x.index)
         sddsdata.SetFixedRowCountMode(x.index)
         #define parameters
         if sddsdata.DefineSimpleParameter(x.index, "ParameterA", "mm", x.SDDS_DOUBLE) != 1:
              raise ValueError
         #define arrays, first is 1 dimensional, second is 2 dimensional
         if sddsdata.DefineSimpleArray(x.index, "ArrayA", "DegC", x.SDDS_DOUBLE, 1) != 1:
              raise ValueError
         if sddsdata.DefineSimpleArray(x.index, "ArrayB", "DegC", x.SDDS_DOUBLE, 2) != 1:
              raise ValueError
         #define columns
         if sddsdata.DefineSimpleColumn(x.index, "ColumnA", "Volts", x.SDDS_DOUBLE) != 1:
              raise ValueError
         if sddsdata.DefineSimpleColumn(x.index, "ColumnB", "Amps", x.SDDS_DOUBLE) != 1:
              raise ValueError
         #write SDDS header
         if sddsdata.WriteLayout(x.index) != 1:
              raise ValueError
         #start SDDS page, allocate 2 rows. This means we have to flush the data every 2 steps.
         if sddsdata.StartPage(x.index, 2) != 1:
              raise ValueError
         #set parameter values
         if sddsdata.SetParameter(x.index, "ParameterA", 1.1) != 1:
              raise ValueError
         #set array values
         if sddsdata.SetArray(x.index, "ArrayA", [1, 2, 3], [3]) != 1:
              raise ValueError
         if sddsdata.SetArray(x.index, "ArrayB", [1, 2, 3, 4, 5, 6], [2, 3]) != 1:
              raise ValueError
         #set all columns, one row at a time
         if sddsdata.SetRowValues(x.index, 0, ["ColumnA", 1, "ColumnB", 1]) != 1:
              raise ValueError
         if sddsdata.SetRowValues(x.index, 1, ["ColumnA", 2, "ColumnB", 2]) != 1:
              raise ValueError
         #update page because we readed the row allocation limit set in the StartPage command
         if sddsdata.UpdatePage(x.index, x.SDDS_FLUSH_TABLE) != 1:
              raise ValueError
         #set more rows
         if sddsdata.SetRowValues(x.index, 2, ["ColumnA", 3, "ColumnB", 3]) != 1:
              raise ValueError
         #update page
         if sddsdata.UpdatePage(x.index, x.SDDS_FLUSH_TABLE) != 1:
              raise ValueError
         
         #close SDDS output file
         if sddsdata.Terminate(x.index) != 1:
              raise ValueError

     except:
          sddsdata.PrintErrors(x.SDDS_VERBOSE_PrintErrors)
          raise
     
def demo5(output):
     """Open an existing SDDS file and add rows to the last page without loading in the whole file into memory."""

     x = SDDS(0)

     try:
         #open SDDS output file
         rows = sddsdata.InitializeAppendToPage(x.index, output, 100);
         if rows == 0:
              raise ValueError
         #set all columns, one row at a time
         if sddsdata.SetRowValues(x.index, rows, ["ColumnA", 4, "ColumnB", 4]) != 1:
              raise ValueError
         if sddsdata.SetRowValues(x.index, rows+1, ["ColumnA", 5, "ColumnB", 5]) != 1:
              raise ValueError
         if sddsdata.SetRowValues(x.index, rows+2, ["ColumnA", 6, "ColumnB", 6]) != 1:
              raise ValueError
         #update page
         if sddsdata.UpdatePage(x.index, x.SDDS_FLUSH_TABLE) != 1:
              raise ValueError
         #we could start another page here if we wanted to
         
         #close SDDS output file
         if sddsdata.Terminate(x.index) != 1:
              raise ValueError

     except:
          sddsdata.PrintErrors(x.SDDS_VERBOSE_PrintErrors)
          raise
     
def demo6(output):
     """Open an existing SDDS file and add a new page."""

     x = SDDS(0)

     try:
         #open SDDS output file
         if sddsdata.InitializeAppend(x.index, output) != 1:
              raise ValueError
         #Allocate rows
         if sddsdata.StartPage(x.index, 100) != 1:
              raise ValueError
         #set parameter values
         if sddsdata.SetParameter(x.index, "ParameterA", 1.1) != 1:
              raise ValueError
         #set array values
         if sddsdata.SetArray(x.index, "ArrayA", [1, 2, 3], [3]) != 1:
              raise ValueError
         if sddsdata.SetArray(x.index, "ArrayB", [1, 2, 3, 4, 5, 6], [2, 3]) != 1:
              raise ValueError
         #set all columns, one row at a time
         if sddsdata.SetRowValues(x.index, 0, ["ColumnA", 7, "ColumnB", 7]) != 1:
              raise ValueError
         if sddsdata.SetRowValues(x.index, 1, ["ColumnA", 8, "ColumnB", 8]) != 1:
              raise ValueError
         if sddsdata.SetRowValues(x.index, 2, ["ColumnA", 9, "ColumnB", 9]) != 1:
              raise ValueError
         #update page
         if sddsdata.WritePage(x.index) != 1:
              raise ValueError
         #we could start another page here if we wanted to
         
         #close SDDS output file
         if sddsdata.Terminate(x.index) != 1:
              raise ValueError

     except:
          sddsdata.PrintErrors(x.SDDS_VERBOSE_PrintErrors)
          raise
     
