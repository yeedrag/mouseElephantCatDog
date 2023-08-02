- block.js
	- class TypeManager
- .constructor(typeList)
		- .addType(type, typeSpec)
			- type: string
			- typeSpec: object
				- readableName: string
				- description: string
				- output: object
					- type: string.
					- description: string
				- input: object 
					- type: string. The output datatype should be the same as the input datatype
					- checker: function
					- description: string
				- args: [{ name, type, possibleValues, checker }]
		- .getType()
		- .listTypes()
	- class Block extends EventTarget
		- .constructor({typeManager, source})
		- .type -- readonly
		- .args
		<!-- - .inputType（其實我不知道這個東西有沒有需要 type） -->
		<!-- - .outputType（同上） -->
		- .exportObj() // 目前構想是只把上面的部分匯出

		// 然後是一些附在 BlockManager 上才能操作的屬性
		- .pos
			- 一個 Object，有 .x, .y
			- 小魔術：可以用 = [number, number] 快速設定
		- .id -- readonly
		- ._assignManager({manager, id, element})
			- 用途：給 BlockManager 的 addBlock 用
			- 設定 manager, id，並且在 element 裡面加上 div.inputPorts, div.header, div.content, div.outputPorts 還有做好設定
		- .manager -- readonly
		- .element
		- .remove()
			- 呼叫 .manager，把自己刪掉
			- 呼叫所有有關聯的 Line，把他們刪掉
			- 把 .manager, .pos, .in/outputlines 刪掉
			- ……但是，這個 block 依然可以再次被加入 manager，所以可被重新使用

		// 接著是一些跟 Line 有關的特性
		- ._assignInputLine(line, index)
			- 把 line 的 outputPort 附加到 this.inputPortsEl 的第 index 位
			- 把 line 塞到 this.inputLines 的第 index 個
		- ._unassignInputLines(line)
		- .inputLines -- readonly
		- ._assignOutputLine(line, index)
		- ._unassignOutputLine(line, index)
		- .outputLines -- readonly
		- event: `change`
			- e.field = "pos" | "inputPorts" | "outputPorts" 之類的
		- ._assignManager({manager, id, element})
		- ._unassignManager()

		// 這個用來快速匯出，給 BlockManager 用的
		- ._exportObjForManager()

	- class Line
		- .constructor(blockManager)
			- 指定 manager（block 的那些 id 的根據）
			- 把 .element 塞入 blockManager.ws
			- 生成 .inputPort 跟 .outputPort，供 BlockManager 或 Block 操作
			- 生成 .element
		- .manager // readonly

		// 一些 element
		- .inputPort，是個 div.port
		- .outputPort，是個 div.port
		- .element，是個 svg

		// 由 BlockManager 管理
		- _setInputBlock(blockId, index)
			- 先解除舊的 blockId
			- 設定 inputBlockId
			- 從 manager 中找到相應 id 的 Block，然後 call 它的 ._assignInputLine(blockId, index)
		- .inputBlockId -- readonly
		- .inputBlock -- readonly
			- 跟 manager 求有該 id 的 block

		- _setOutputBlock(blockId, index)
		- .outputBlockId -- readonly
		- .outputBlock -- readonly

		- .redraw()
		- .remove()
			- 解除 .in/outputBlock
			- 把 .element 刪掉
			- 在 .manager 中刪掉自己
			- ……這條 Line 不可被重新使用，因為 manager 不能重新設定
	- class BlockManager extends EventTarget
		- constructor(wsContainer, source)
		- .ws // workspaceElement
		- .wsC // workspaceContainer
		- .wsCfg // workspaceConfiguration
		- .addBlock(block)
		- .blocks // readonly
		- .removeBlockById(id)
		- .genLine() // return a Line
		- ._removeLine() // for Line
		- .export() // return json
