/**
 * @param {JSON} blockParameterData
 */
export function getParameters(blockParameterData, idNum, type) {
    const Bar = document.getElementById('PBarID:' + idNum);
    const numericParameters = blockParameterData[type]["numeral"]
    for (let numericParameter in numericParameters) {
        const paraText = document.createElement('div');
        paraText.textContent = numericParameter;
        Bar.appendChild(paraText);

        const inputBox = document.createElement('input');
        inputBox.setAttribute('type', 'text');
        inputBox.setAttribute('id', );

    }

}
