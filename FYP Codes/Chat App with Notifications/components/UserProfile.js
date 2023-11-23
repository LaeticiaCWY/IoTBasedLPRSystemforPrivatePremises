import React, { useState } from "react";
import { useAuthState } from "react-firebase-hooks/auth";
import { auth } from "../firebase";
import { db } from "../firebase"; // Assuming you have a Firebase Firestore database

const UserProfile = () => {
  const [user] = useAuthState(auth);
  const [newDisplayName, setNewDisplayName] = useState("");
  
  const handleChangeName = () => {
    if (newDisplayName) {
      db.collection("users").doc(user.uid).update({
        displayName: newDisplayName,
      });
    }
  };

  return (
    <div>
      <h2>Change Display Name</h2>
      <input
        type="text"
        placeholder="New Display Name"
        value={newDisplayName}
        onChange={(e) => setNewDisplayName(e.target.value)}
      />
      <button onClick={handleChangeName}>Change Name</button>
    </div>
  );
};

export default UserProfile;
