import {hello} from './something.js'

window.hello = hello;
//I don't know how to avoid using all the function in global scope. all items with "window." prefix should be edited in some way.

//initailize ans util 
var idMax = -1;
function getIdNumber(string) {
    var idNum=0;
    for(let i = string.length-1, k = 1;string[i]!=':';i--) {
        idNum+=k*parseInt(string[i]);
        k*=10;
    }
    return idNum;
}
function test() {
    console.log("test");
}
window.test = test;
//initailize 

//drag and drop--------------------------------------------------------------------------------------------------------------------------

function drag_start(event) {
    var style = window.getComputedStyle(event.target, null);
    var idNum = getIdNumber(event.target.id);
    event.dataTransfer.setData("text/plain",
    (parseInt(style.getPropertyValue("left"),10) - event.clientX) + ',' + (parseInt(style.getPropertyValue("top"),10) - event.clientY)+','+idNum);
    return false;
} 
window.drag_start = drag_start;

function drag_over(event) { 
    event.preventDefault(); 
    return false; 
} 
window.drag_over = drag_over;

function drop(event) { 
    var offset = event.dataTransfer.getData("text/plain").split(',');
    var dm = document.getElementById('blockID:'+offset[2]);
    dm.style.left = (event.clientX + parseInt(offset[0],10)) + 'px';
    dm.style.top = (event.clientY + parseInt(offset[1],10)) + 'px';
    event.preventDefault();
    return false;
} 
window.drop = drop;

//drag and drop--------------------------------------------------------------------------------------------------------------------------
//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------
function openParameterBar() {
    console.log("test");
    document.getElementById("PBarID:1").style.width = "250px";
    //document.getElementById("main").style.marginLeft = "250px";
}
window.openParameterBar = openParameterBar;

function closeParameterBar() {
    document.getElementById("PBarID:1").style.width = "0";
    //document.getElementById("main").style.marginLeft= "0";
}
window.closeParameterBar = closeParameterBar;
//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------

//add new block----------------------------------------------------------------------------------------------------------------------------
function addBlock() {
    const ele = document.createElement("div");
    idMax++;
    ele.setAttribute('id', 'blockID:' + idMax);
    ele.setAttribute('class', 'block');
    ele.setAttribute('draggable', true);
    ele.setAttribute('ondragstart', "drag_start(event)");

    const type = document.getElementById('addBlock-modal-selecter').value;
    console.log(type);
    ele.setAttribute('blockType', type);
    const text = document.createElement("h2");
    text.textContent = type;
    ele.appendChild(text);
    const closeButton = document.createElement("a");
    closeButton.setAttribute('href', 'javascript:void(0)');
    closeButton.setAttribute('id', 'closeButton:' + idMax);
    closeButton.setAttribute('onclick','deleteBlock(event)');
    closeButton.textContent = "X";
    ele.appendChild(closeButton);

    const workspace = document.getElementById('workSpace');
    workspace.appendChild(ele);
}
window.addBlock = addBlock;
//add new block----------------------------------------------------------------------------------------------------------------------------

//delete block----------------------------------------------------------------------------------------------------------------------------
function deleteBlock(event) {
    var idNum = getIdNumber(event.target.id);
    console.log(idNum);
    
    document.getElementById('blockID:' + idNum).remove();
    for(let i = idNum+1;i <= idMax; i++) {
        let cur = document.getElementById('blockID:' + i);
        cur.setAttribute('id', 'blockID:' + (i-1));
    }
    idMax--;
    console.log(idMax);
    
}
window.deleteBlock = deleteBlock;

//delete block----------------------------------------------------------------------------------------------------------------------------
