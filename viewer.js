var numDimensions;
var dimensions = [];
var images = [];

function loadIdx(filename) {
  var req = new XMLHttpRequest();
  req.onload = function() { populateTable(req.response); }
  req.responseType = "arraybuffer";
  req.open("GET", filename, true);
  req.send();
}

function populateTable(rawIdx) {
  var idx = new Uint8Array(rawIdx);
  numDimensions = idx[3];
  for (var i = 0; i < numDimensions; i++) {
    dimension = intFromBigEndianByteArray(idx.subarray(4 + 4*i, 8 + 4*i));
    dimensions.push(dimension);
  }
  constructImages(idx);
  createDynamicViewer(0);
  loadStaticComparisons("incorrect_predictions");
}

function constructImages(idx) {
  var counter = 4 + 4*numDimensions;
  for (var i = 0; i < dimensions[0]; i++) {
    var image = [];
    for (var j = 0; j < dimensions[1]; j++) {
      var row = [];
      for (var k = 0; k < dimensions[2]; k++) {
        row.push(idx[counter++]);
      }
      image.push(row);
    }
    images.push(image);
  }
}

function loadStaticComparisons(filename) {
  var req = new XMLHttpRequest();
  req.onload = function() {
    createStaticComparisons(JSON.parse(req.responseText));
  }
  req.open("GET", filename, true);
  req.send();
}

function createStaticComparisons(incorrectPredictions) {
  document.getElementById("status").innerHTML = "Loading incorrect predictions...";
  setTimeout(function() {
    for (var i = 0; i < incorrectPredictions.length; i++) {
      prediction = incorrectPredictions[i];
      createStaticViewer(prediction[0], prediction[1], prediction[2]);
    }
    document.getElementById("status").innerHTML = ""
  }, 0);
}

function createDynamicViewer(index) {
  var table = createTable();
  showImage(index, table);
  document.body.appendChild(table);
  var input = createInputForTable(table);
  input.value = index;
  document.body.appendChild(input);
  input.focus();
}

function createStaticViewer(index, actualLabel, predictedLabel) {
  var table = createTable();
  showImage(index, table);
  document.body.appendChild(table);
  var p = document.createElement("p");
  p.innerHTML = "predicted " + predictedLabel + ", actual " + actualLabel;
  document.body.appendChild(p);
}

function createTable() {
  var table = document.createElement("table");
  var numRows = dimensions[1];
  var numCols = dimensions[2];
  for (var i = 0; i < numRows; i++) {
    var tr = table.insertRow();
    for (var j = 0; j < numCols; j++) {
      tr.insertCell();
    }
  }
  return table;
}

function createInputForTable(table) {
  var input = document.createElement("input");
  input.type = "number";
  input.onchange = function() { showImage(input.value, table) }
  return input
}

function showImage(index, table) {
  var image = images[index];
  var numRows = dimensions[1];
  var numCols = dimensions[2];
  for (var i = 0; i < numRows; i++) {
    var tr = table.rows[i];
    var row = image[i];
    for (var j = 0; j < numCols; j++) {
      var cellString = Number(255 - image[i][j]).toString(16);
      if (cellString.length == 1) { cellString = "0" + cellString; }
      tr.cells[j].style.backgroundColor = "#" + cellString + cellString + cellString;
    }
  }
}

function intFromBigEndianByteArray(bytes) {
  return bytes[0]*Math.pow(2,24) + bytes[1]*Math.pow(2,16) + bytes[2]*Math.pow(2,8) + bytes[3];
}

