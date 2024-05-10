import React, { useRef, useState, useEffect } from 'react';
import './ChatComponentStyle.css';

import Switch from "react-switch";
import inputData from './forecasted_values_month.json'
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';

const ChatComponent = () => {
    const [chatHistory, setChatHistory] = useState([{ role: 'bot', content: "Hello! I'm here to assist you with any questions you may have." }]);
    const [userMessage, setUserMessage] = useState('');
    const inputRef = useRef(null);
    const chatContainerRef = useRef(null);
    const [appliances, setAppliances] = useState({
        fridge: 'off',
        furnace: 'off',
        dishwasher: 'off'
    });
    const [currentTime, setCurrentTime] = useState(0);

    const [chartValues , setChartValue] = useState(inputData);
     useEffect(()=>{
        // console.log(chartValues)
     },[chartValues])

    const [appliancesKWHValues, setAppliancesKWHValues] = useState({
        fridge: 0,
        furnace: 0,
        dishwasher: 0
    });

    useEffect(()=>{
        sendData()
    },[appliancesKWHValues])
    
    const sendData = () => {
        const dataToSend = {
            currentTime: currentTime,
            currentValues: appliancesKWHValues,
            predictedValues: inputData
        };
    
        fetch('http://localhost:8000/anomalyDetection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(dataToSend)
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = data.response;
            checkBotResponse(botResponse);
            console.log(data);
        })
        .catch(error => {
            console.error(error);
        });
    };
    
    // Call the fetchData function every minute
    // const interval = setInterval(sendData, 60000);

    const sendMessage = () => {
        const messageInput = userMessage.trim();
        if (!messageInput) return;

        const message = `
        Answer the above question based on the following format:
        
        user prompt: "Tell the current status of all the appliances"
        formatted output: {"to_say" : "Here is the current status of all appliances",  "service" : "status()", "target" : "all"}
      
        ` + "user prompt: " + '"' + messageInput + '"';
    
        const userMessageObj = { role: 'user', content: messageInput };
        setChatHistory(prevChatHistory => [...prevChatHistory, userMessageObj]);
        setUserMessage('');

        inputRef.current.value = '';
        inputRef.current.focus();
    
        fetch("http://localhost:8000/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: message
            })
        })
        .then((response) => response.json())
        .then((data) => {
            const botResponse = data.response;
            const ans = checkBotResponse(botResponse);
            const botMessageObj = { role: 'bot', content: ans };
            setChatHistory(prevChatHistory => [...prevChatHistory, botMessageObj]);
        })
        .catch((error) => console.error(error));
    }

    const checkBotResponse = (botResponse) => {
        console.log(botResponse)
        botResponse = botResponse.replace("formatted output: ", "");
        const data = JSON.parse(botResponse);

        const message = data.to_say;
        const service = data.service;
        const target = data.target;
        if (service === "anomaly()") {
            const botMessageObj = { role: 'bot', content: message };
            setChatHistory(prevChatHistory => [...prevChatHistory, botMessageObj]);
        }
            
        if (target === "fridge") {
            setAppliances(prevAppliances => ({ ...prevAppliances, fridge: service === "turn_on()" ? 'on' : 'off' }));
        } else if (target === "furnace") {
            setAppliances(prevAppliances => ({ ...prevAppliances, furnace: service === "turn_on()" ? 'on' : 'off' }));
        } else if (target === "dishwasher") {
            setAppliances(prevAppliances => ({ ...prevAppliances, dishwasher: service === "turn_on()" ? 'on' : 'off' }));
        }
        if (target === "all") {
            // Handle target === "all"
        }
        return message;
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    }

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory]);

    const setKWHValue=(value,appliance)=>{
        if (appliance === "fridge") {
            setAppliancesKWHValues(prevAppliancesKWHValues => ({ ...prevAppliancesKWHValues, fridge: value }));
        } else if (appliance === "furnace") {
            setAppliancesKWHValues(prevAppliancesKWHValues => ({ ...prevAppliancesKWHValues, furnace: value }));
        } else if (appliance === "dishwasher") {
            setAppliancesKWHValues(prevAppliancesKWHValues => ({ ...prevAppliancesKWHValues, dishwasher: value }));
        }
    }
    
    const timerFunction = () => {
        // console.log('Timer function called!');
        setCurrentTime(prevTime => prevTime + 1);
      };

      useEffect(()=>{
        setChartValue(prevChartValues => {
              const updatedFridgeData = [...prevChartValues['fridge_current']];
              updatedFridgeData[currentTime] = appliances['fridge']==='on'?parseFloat(appliancesKWHValues['fridge']):0;
              const updatedFurnaceData = [...prevChartValues['furnace_current']];
              updatedFurnaceData[currentTime] = appliances['furnace']==='on'?parseFloat(appliancesKWHValues['furnace']):0;
              const updatedDishwasherData = [...prevChartValues['dishwasher_current']];
              updatedDishwasherData[currentTime] = appliances['dishwasher']==='on'?parseFloat(appliancesKWHValues['dishwasher']):0;
              return {
                fridge_predicted: prevChartValues['fridge_predicted'],
                fridge_current: updatedFridgeData,
                dishwasher_predicted: prevChartValues['dishwasher_predicted'],
                dishwasher_current: updatedDishwasherData,
                furnace_predicted: prevChartValues['furnace_predicted'],
                furnace_current: updatedFurnaceData
              };
            });
      },[currentTime])
    
      useEffect(() => {
        // Set up an interval to call the timerFunction every 60 seconds- 60000
        const intervalId = setInterval(timerFunction, 5000);
    
        // Clean up the interval on component unmount
        return () => clearInterval(intervalId);
      }, []); 

    const handleSwitchChange = (appliance) => {
        setAppliances(prevAppliances => ({
            ...prevAppliances,
            [appliance]: prevAppliances[appliance] === 'on' ? 'off' : 'on'
        }));
    };

    return (
        <div className='row justify-content-center'>
            <div className='col' style={{ height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className='row justify-content-center' style={{ height: '40vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', marginBottom: '10px', padding: '10px', textAlign: 'center' }}>
                    Appliances - TIME : {currentTime}
                    <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-around', width: '100%' }}>
                        {Object.keys(appliances).map((appliance) => (
                            <div key={appliance} style={{ borderRadius:10,flex: 1, border: '1px solid grey', margin: '5px', height: '20vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                                <h6>{appliance.toUpperCase()}</h6>
                                <Switch
                                    onChange={() => handleSwitchChange(appliance)}
                                    checked={appliances[appliance] === 'on'}
                                    checkedIcon={<h6 style={{ margin:0, marginLeft:3, color:'white', paddingTop:3}}>On</h6>}
                                    uncheckedIcon={<h6 style={{ margin:0, marginLeft:2, color:'white', paddingTop:3}}>Off</h6>}
                                />
                                {/* checked={appliances[appliance]==='on'?true:false} 
                                checkedIcon={<h6 style={{ margin:0, marginLeft:3, color:'white', paddingTop:3}}>On</h6>}
                                uncheckedIcon={<h6 style={{ margin:0, marginLeft:2, color:'white', paddingTop:3}}>Off</h6>}/> */}
                                {appliances[appliance]==='on'?<input
                                key={appliances[appliance]}
                                style={{width:'30%', marginTop:10, border: '1px solid black',borderRadius:3, textAlign:'center'}}
                                    type="text"
                                    value={appliancesKWHValues[appliance]} // Controlled input: value is controlled by state
                                    onChange={(event)=>{setKWHValue(event.target.value,appliance)}} // Event handler for input change
                                />:<></>}
                            </div>
                        ))}
                    </div>
                </div>
                <div className='row justify-content-center' style={{ height: '48vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', padding: '10px', textAlign: 'center' }}>
                    <div id={currentTime} style={{ height: '100%', width:'100%'}}>
                        <HighchartsReact key={currentTime}
                            highcharts={Highcharts}
                            options={{
                            title: {
                                text: 'Chart',
                            },
                            xAxis: {
                                categories: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
                            },
                            yAxis: {
                                title: {
                                text: 'Value',
                                },
                            },
                            series: [
                                {
                                name: 'Fridge-Predicted',
                                data: chartValues['fridge_predicted'],
                                },
                                {
                                name: 'Fridge-Current',
                                data: chartValues['fridge_current'],
                                },
                                {
                                name: 'Furnace-Predicted',
                                data: chartValues['furnace_predicted'],
                                },
                                {
                                name: 'Furnace-Current',
                                data: chartValues['furnace_current'],
                                },
                                {
                                name: 'Dishwasher-Predicted',
                                data: chartValues['dishwasher_predicted'],
                                },
                                {
                                name: 'Dishwasher-Current',
                                data: chartValues['dishwasher_current'],
                                }
                            ],
                            }}
                        />
                        </div>
                </div>
            </div>
            <div className='col' style={{ height: '98vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', display: 'flex', flexDirection: 'column', alignItems: 'center', paddingLeft: '0' }}>
                {/* chatbot */}
                <div ref={chatContainerRef} style={{ flex: 1, border: '1px solid rgba(128, 128, 128, 0.5)', width: '100%', overflowY: 'auto', textAlign: 'left', padding: '10px' }}>
                    {chatHistory.map((message, index) => (
                        <p key={index}><strong>{message.role === 'bot' ? 'Chatbot' : 'You'}:</strong> {message.content}</p>
                    ))}
                </div>
                <div className='row' style={{ width: '100%', paddingTop: '5px' }}>
                    <div className='col-11' style={{ paddingLeft: '0' }}>
                        <input
                            type="text"
                            placeholder="Type your message..."
                            className='form-control'
                            ref={inputRef}
                            onChange={(e) => setUserMessage(e.target.value)}
                            onKeyPress={handleKeyPress}
                        />
                    </div>
                    <div className='col-1' style={{ paddingRight: '0', paddingLeft: '0' }}>
                        <button className='btn btn-primary' onClick={sendMessage}>Send</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatComponent;
