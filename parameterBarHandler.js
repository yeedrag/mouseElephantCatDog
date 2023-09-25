/**
 * @param {JSON} blockParameterData
 */
export function getParameters(newBar ,blockParameterData, idNum, blockType) {
    const numericParameters = blockParameterData[blockType]["numeral"];
    for (let numericParameter in numericParameters) {
        const paraText = document.createElement('div');
        paraText.textContent = numericParameter;
        newBar.appendChild(paraText);

        const inputBox = document.createElement('input');
        inputBox.setAttribute('type', 'text');
        inputBox.setAttribute('id',numericParameter + ':' + idNum);
        newBar.appendChild(inputBox);

        const cutLine = document.createElement('hr');
        cutLine.setAttribute('class', 'parameterCutLine');
        newBar.appendChild(cutLine);
    }


    const selectableParameters = blockParameterData[blockType]["selectable"];
    for (let selectableParameter in selectableParameters) {
        const paraText = document.createElement('div');
        paraText.textContent = selectableParameter;
        newBar.appendChild(paraText);
        
    }


}
