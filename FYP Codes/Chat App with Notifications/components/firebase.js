import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getMessaging, getToken, onMessage } from "firebase/messaging"; // Correct import

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCZQHJ4_9A3fEz6vPiHk6iBoW7JBLr_rnA",
  authDomain: "react-chat-e54f8.firebaseapp.com",
  projectId: "react-chat-e54f8",
  storageBucket: "react-chat-e54f8.appspot.com",
  messagingSenderId: "1090126814076",
  appId: "1:1090126814076:web:b99b39e9fa514a3c6922f9",
  measurementId: "G-M9HQK3SB77"
};
// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const messaging = getMessaging(app); // Initialize Firebase Cloud Messaging

export const initializeFirebaseMessaging = async () => {
  try {
    const currentToken = await getToken(messaging);
    if (currentToken) {
      // Send the token to your server and store it if needed
      return currentToken;
    } else {
      // Handle the case when no token is available
      console.log('No registration token available.');
      return null;
    }
  } catch (error) {
    console.error('Error retrieving FCM token:', error);
    throw error;
  }
};

export const onMessageListener = () =>
  new Promise((resolve) => {
    onMessage(messaging, (payload) => {
      resolve(payload);
    });
});