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



export const requestPermission = (messaging) => {
  console.log("Requesting User Permission......");
  Notification.requestPermission().then((permission) => {
    if (permission === "granted") {
      console.log("Notification User Permission Granted.");
      return getToken(messaging, {
        vapidKey: `BDUVPLcbyG6gnwTB9kZFWXf3iDkzkf-jY7plYoKtRHi0q-KFLr4uWEjcSXePfnNC7Np25hAPhR7SkEaTzq_qDFw`,
      })
        .then((currentToken) => {
          if (currentToken) {
            console.log('Client Token: ', currentToken);
          } else {
            console.log('Failed to generate the app registration token.');
          }
        })
        .catch((err) => {
          console.log('An error occurred when requesting to receive the token.', err);
        });
    } else {
      console.log("User Permission Denied.");
    }
  });
};
