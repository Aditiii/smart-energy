import React, { useRef, useState, useEffect } from 'react';
import './ChatComponentStyle.css';

import Switch from "react-switch";
import inputData from './forecasted_values_month.json'
import { ResponsiveLine } from '@nivo/line'

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
    const [chartData,setChartData] = useState([]);
     useEffect(()=>{
        const result = Object.entries(chartValues).map(([key, values], index) => ({
            id: key,
            color: `hsl(${index * 40}, 70%, 50%)`,
            data: values.map((value, index) => ({ x: index, y: value }))
          }));
          console.log(currentTime, result);
          setChartData(result)
     },[chartValues, currentTime])

    const [appliancesKWHValues, setAppliancesKWHValues] = useState({
        fridge: 0,
        furnace: 0,
        dishwasher: 0
    });

    const sendData = () => {
        fetch('http://localhost:8000/anomalyDetection', {
            
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
        })
        .catch(error => {
            console.error(error);
        });
    };
    
    // Call the fetchData function every minute
    const interval = setInterval(sendData, 60000);

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
        console.log('Timer function called!');
        setCurrentTime(prevTime => prevTime + 1);
      };

      useEffect(()=>{
        if (appliances['fridge'] === 'on') {
            setChartValue(prevChartValues => {
              const updatedFridgeData = [...prevChartValues['fridge']];
              updatedFridgeData[currentTime] = appliancesKWHValues['fridge'];
              return {
                ...prevChartValues,
                fridge: updatedFridgeData
              };
            });
          }
        
      },[currentTime])
    
      useEffect(() => {
        // Set up an interval to call the timerFunction every 60 seconds- 60000
        const intervalId = setInterval(timerFunction, 5000);
    
        // Clean up the interval on component unmount
        return () => clearInterval(intervalId);
      }, []); 

    return (
        <div className='row justify-content-center'>
            <div className='col' style={{ height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className='row justify-content-center' style={{ height: '50vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', marginBottom: '10px', padding: '10px', textAlign: 'center' }}>
                    Appliances - TIME : {currentTime}
                    <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-around', width: '100%' }}>
                        {Object.keys(appliances).map((appliance) => (
                            <div key={appliance} style={{ borderRadius:10,flex: 1, border: '1px solid grey', margin: '5px', height: '20vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                                <h6>{appliance.toUpperCase()}</h6>
                                <Switch onChange={()=>{}} 
                                checked={appliances[appliance]=='on'?true:false} 
                                checkedIcon={<h6 style={{ margin:0, marginLeft:3, color:'white', paddingTop:3}}>On</h6>}
                                uncheckedIcon={<h6 style={{ margin:0, marginLeft:2, color:'white', paddingTop:3}}>Off</h6>}/>
                                {appliances[appliance]=='on'?<input
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
                <div className='row justify-content-center' style={{ height: '50vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', padding: '10px', textAlign: 'center' }}>
                    Chart
                    <div id={currentTime} style={{ height: '100%', width:'100%'}}>
                        { chartData && chartData.length>0 ?<ResponsiveLine   
                                key={Date.now()}                        
                                data={chartData}
                                margin={{ top: 20, right: 150, bottom: 50, left: 60 }}
                                xScale={{ type: 'point' }}
                                yScale={{
                                    type: 'linear',
                                    min: 'auto',
                                    max: 'auto',
                                    stacked: true,
                                    reverse: false
                                }}
                                yFormat=" >-.2f"
                                axisTop={null}
                                axisRight={null}
                                axisBottom={null}
                                axisLeft={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                    legend: '',
                                    legendOffset: -40,
                                    legendPosition: 'middle',
                                    truncateTickAt: 0
                                }}
                                colors={{ scheme: 'category10' }}
                                pointColor={{ from: 'color', modifiers: [] }}
                                pointSize={5}
                                pointBorderWidth={2}
                                pointBorderColor={{ from: 'serieColor' }}
                                pointLabel="data.yFormatted"
                                pointLabelYOffset={-12}
                                enableTouchCrosshair={true}
                                useMesh={true}
                                legends={[
                                    {
                                        anchor: 'bottom-right',
                                        direction: 'column',
                                        justify: false,
                                        translateX: 100,
                                        translateY: 0,
                                        itemsSpacing: 0,
                                        itemDirection: 'left-to-right',
                                        itemWidth: 80,
                                        itemHeight: 20,
                                        itemOpacity: 0.75,
                                        symbolSize: 12,
                                        symbolShape: 'circle',
                                        symbolBorderColor: 'rgba(0, 0, 0, .5)',
                                        effects: [
                                            {
                                                on: 'hover',
                                                style: {
                                                    itemBackground: 'rgba(0, 0, 0, .03)',
                                                    itemOpacity: 1
                                                }
                                            }
                                        ]
                                    }
                                ]}
                            />:<></>}
                        </div>
                </div>
            </div>
            <div className='col' style={{ height: '100vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', display: 'flex', flexDirection: 'column', alignItems: 'center', paddingLeft: '0' }}>
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
