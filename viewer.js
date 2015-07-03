var numDimensions;
var dimensions = [];
var images = [];

function loadIdx() {
  var req = new XMLHttpRequest();
  req.onload = function() { populateTable(req.response); }
  req.responseType = "arraybuffer";
  req.open("GET", "train-images-idx3-ubyte", true);
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
  createTable();
  showImage(0);
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

function createTable() {
  var table = document.getElementById("table");
  var numRows = dimensions[1];
  var numCols = dimensions[2];
  for (var i = 0; i < numRows; i++) {
    var tr = table.insertRow();
    for (var j = 0; j < numCols; j++) {
      tr.insertCell();
    }
  }
  table.style.visibility = "visible";
}

function showImage(index) {
  var image = images[index];
  var table = document.getElementById("table");
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

