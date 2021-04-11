import React, { useEffect, useRef } from 'react'
import {
    ChakraProvider,
    Box,
    Text,
    Link,
    VStack,
    Code,
    Grid,
    theme,
    Container,
    Center,
} from '@chakra-ui/react';

export default function MainPage() {
    const videoRef = useRef();
    //good
    useEffect(() => {
        navigator.mediaDevices
            .getUserMedia({
                video: true
            })
            .then((stream) => {
                videoRef.current.srcObject = stream
            })
    })
    return (
        <Container>
            <Center>
                <video
                    autoPlay
                    ref={videoRef}
                    style={{ width: 600 }}></video>
            </Center>


        </Container>
    )
}
