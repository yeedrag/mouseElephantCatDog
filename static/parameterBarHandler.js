



/**
 * @param {JSON} blockParameterData
 */

//shit the function is ugly
export function getParameters(newBar ,blockParameterData, idNum, blockType, backendOutput, isActivation) {

    let BlkArgument = {};
    if(isActivation == true) BlkArgument["blockName"] = 'Activation';
    else BlkArgument["blockName"] = blockType;
    BlkArgument["parent"] = [];
    BlkArgument["child"] = [];
    BlkArgument["args"] = {};
    if(isActivation == true) BlkArgument["args"]["mode"] = blockType;
    BlkArgument["args"]["inputSize"] = [];
    BlkArgument["args"]["outputSize"] = [];
    //a temporary area for testing-------------------------------------------------------------------------------------------------- 
    const parentText = document.createElement('div');
    parentText.textContent = 'parent';
    newBar.appendChild(parentText);
    const parentInput = document.createElement('input');
    parentInput.setAttribute('type', 'text');
    parentInput.setAttribute('id', 'parent' + ':' + idNum);
    parentInput.setAttribute('onchange', 'onInputChange(event)');
    newBar.appendChild(parentInput);
    const childText = document.createElement('div');
    childText.textContent = 'child';
    newBar.appendChild(childText);
    const childInput = document.createElement('input');
    childInput.setAttribute('type', 'text');
    childInput.setAttribute('id', 'child' + ':' + idNum);
    childInput.setAttribute('onchange', 'onInputChange(event)');
    newBar.appendChild(childInput);
    //a temporary area for testing-------------------------------------------------------------------------------------------------- 

    const numericParameters = blockParameterData[blockType]["numeral"];
    for (let numericParameter in numericParameters) {
        const paraText = document.createElement('div');
        paraText.textContent = numericParameter;
        newBar.appendChild(paraText);

        const inputBox = document.createElement('input');
        inputBox.setAttribute('type', 'text');
        inputBox.setAttribute('id',numericParameter + ':' + idNum);
        inputBox.setAttribute('onchange', 'onInputChange(event)');
        newBar.appendChild(inputBox);

        const cutLine = document.createElement('hr');
        cutLine.setAttribute('class', 'parameterCutLine');
        newBar.appendChild(cutLine);

        BlkArgument["args"][numericParameter] = [];
    }

    const selectableParameters = blockParameterData[blockType]["selectable"];
    for (let selectableParameter in selectableParameters) {
        const paraText = document.createElement('div');
        paraText.textContent = selectableParameter;
        newBar.appendChild(paraText);  

        const selectbox = document.createElement('select');
        selectbox.setAttribute('id', selectableParameter + ':' + idNum);
        selectbox.setAttribute('onchange', 'onInputChange(event)');
        const selectValues = blockParameterData[blockType]["selectable"][selectableParameter];
        for(let i in selectValues) {
            const selectOption = document.createElement('option');
            selectOption.setAttribute('value', selectValues[i]);
            selectOption.textContent = selectValues[i];
            selectbox.appendChild(selectOption);
        }
        newBar.appendChild(selectbox);

        const cutLine = document.createElement('hr');
        cutLine.setAttribute('class', 'parameterCutLine');
        newBar.appendChild(cutLine);
        BlkArgument["args"][selectableParameter] = selectValues[0];
    }

    backendOutput.push(BlkArgument);
}

