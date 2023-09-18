//drag and drop--------------------------------------------------------------------------------------------------------------------------
function drag_start(event) {
    var style = window.getComputedStyle(event.target, null);
    event.dataTransfer.setData("text/plain",
    (parseInt(style.getPropertyValue("left"),10) - event.clientX) + ',' + (parseInt(style.getPropertyValue("top"),10) - event.clientY));
    console.log("start");
    return false;
} 
function drag_over(event) { 
    event.preventDefault(); 
    return false; 
} 
function drop(event) { 
    var offset = event.dataTransfer.getData("text/plain").split(',');
    var dm = document.getElementById('blockID:1');
    dm.style.left = (event.clientX + parseInt(offset[0],10)) + 'px';
    dm.style.top = (event.clientY + parseInt(offset[1],10)) + 'px';
    event.preventDefault();
    return false;
} 
//drag and drop--------------------------------------------------------------------------------------------------------------------------
//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------
function test() {
    console.log(sdsl);
}
//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------
function openNav() {
    console.log("test");
    document.getElementById("PBarID:1").style.width = "250px";
    //document.getElementById("main").style.marginLeft = "250px";
}
  
function closeNav() {
    document.getElementById("PBarID:1").style.width = "0";
    //document.getElementById("main").style.marginLeft= "0";
}
