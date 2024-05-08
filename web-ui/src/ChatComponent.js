import React, { useRef, useState, useEffect } from 'react';
import './ChatComponentStyle.css';


const ChatComponent = () => {
    const [chatHistory, setChatHistory] = useState([{ role: 'bot', content: "Hi! I'm here to assist you with any questions you may have." }]);
    const [userMessage, setUserMessage] = useState('');
    const inputRef = useRef(null);
    const chatContainerRef = useRef(null);

    const sendMessage = () => {
        const messageInput = userMessage.trim();
        if (!messageInput) return;

        const message = `
        Answer the above question based on the following format:
        
        user prompt: "Tell the current status of all the appliances"
        formatted output: {"to_say" : "Here is the current status of all appliances",  "service" : "status()", "target" : "all"}
      
        ` + "user prompt: " + '"' + messageInput + '"';
    
        const userMessageObj = { role: 'user', content: userMessage };
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
                question: messageInput
            })
        })
        .then((response) => response.json())
        .then((data) => {
            const botResponse = data.response;
            // const ans = checkBotResponse(botResponse);
            const botMessageObj = { role: 'bot', content: botResponse };
            setChatHistory(prevChatHistory => [...prevChatHistory, botMessageObj]);
        })
        .catch((error) => console.error(error));
    }

    const checkBotResponse = (botResponse) => {
        botResponse = botResponse.replace("formatted output: ", "");
        const data = JSON.parse(botResponse);

        const message = data.to_say;
        const service = data.service;
        const target = data.target;

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

    return (
        <div className='row justify-content-center'>
            <div className='col' style={{ height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className='row justify-content-center' style={{ height: '50vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', marginBottom: '10px', padding: '10px', textAlign: 'center' }}>
                    Appliances
                    <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-around', width: '100%' }}>
                        <div style={{ flex: 1, border: '1px solid grey', margin: '5px', height: '20vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Block 1</div>
                        <div style={{ flex: 1, border: '1px solid grey', margin: '5px', height: '20vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Block 2</div>
                        <div style={{ flex: 1, border: '1px solid grey', margin: '5px', height: '20vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Block 3</div>
                    </div>
                </div>
                <div className='row justify-content-center' style={{ height: '50vh', border: '1px solid rgba(128, 128, 128, 0.5)', width: '50vw', padding: '10px', textAlign: 'center' }}>
                    Chart
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
