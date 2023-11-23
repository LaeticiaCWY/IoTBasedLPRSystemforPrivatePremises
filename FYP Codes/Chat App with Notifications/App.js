// App.js
import React, { useEffect, useState } from "react";
import { auth } from "./firebase"; // Import Firebase configuration
import { useAuthState } from "react-firebase-hooks/auth";
import "./App.css";
import NavBar from "./components/NavBar";
import ChatBox from "./components/ChatBox";
import Welcome from "./components/Welcome";
import NotificationComponent from "./components/NotificationComponent"; // Import the new component
import { initializeFirebaseMessaging } from "./firebase"; // Import the messaging initialization function
import UserProfile from "./components/UserProfile";

function App() {
  const [user] = useAuthState(auth);
  const [notifications, setNotifications] = useState([]); // State variable for notifications

  useEffect(() => {
    // Request permission for browser notifications
    Notification.requestPermission().then(async (permission) => {
      if (permission === "granted") {
        console.log("Notification permission granted.");
        // Retrieve the FCM registration token
        try {
          const token = await initializeFirebaseMessaging(); // Use the messaging initialization function
          console.log("FCM Registration token:", token);
        } catch (error) {
          console.error("Error retrieving FCM token:", error);
        }
      } else {
        console.log("Notification permission denied.");
      }
    });

    // Add an event listener to listen for incoming FCM messages
    navigator.serviceWorker.addEventListener("message", (event) => {
      const { data } = event;
      console.log("FCM Message received:", data);

      // Create a new notification object and add it to the notifications list
      const notification = new Notification(data.notification.title, {
        body: data.notification.body,
      });
      setNotifications((prevNotifications) => [...prevNotifications, notification]);
    });
  }, []);

  return (
    <div className="App">
      <NavBar />
      {!user ? <Welcome /> : <ChatBox />}
      <header className="App-header">
        <NotificationComponent /> {/* Render the new component */}
        <UserProfile /> {/* Render the new component */}
      </header>
    </div>
  );
}

export default App;
